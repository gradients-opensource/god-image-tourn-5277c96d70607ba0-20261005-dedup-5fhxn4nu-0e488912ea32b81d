"""Autograd-safe Flux blocks.

ComfyUI's DoubleStreamBlock/SingleStreamBlock (comfy/ldm/flux/layers.py at the
evaluator's commit) update the residual streams in place (`img += ...`,
`txt += ...`, `x += ...`). In-place updates on tensors that autograd still needs
raise during backward, so training through the stock blocks is impossible.

These forwards compute the identical function with out-of-place adds. Nothing
else changes: same modulation, same attention call, same patch hooks, same
fp16 nan guard. Inference through them is numerically the same as upstream.
"""

import torch


def apply():
    import comfy.ldm.flux.layers as L
    from comfy.ldm.flux.layers import apply_mod, attention

    def double_forward(self, img, txt, vec, pe, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None, transformer_options={}):
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
        patches = transformer_options.get("patches", {})
        extra = transformer_options.copy()

        img_m = apply_mod(self.img_norm1(img), (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_m)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_m = apply_mod(self.txt_norm1(txt), (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_m)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        extra["img_slice"] = [txt.shape[1], q.shape[2]]
        for p in patches.get("attn1_patch", []):
            out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra)
            q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        for p in patches.get("attn1_output_patch", []):
            attn = p(attn, extra)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)
        txt = txt + apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
        txt = txt + apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)
        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
        return img, txt

    def single_forward(self, x, vec, pe, attn_mask=None, modulation_dims=None, transformer_options={}):
        mod = self.modulation(vec)[0] if self.modulation else vec
        patches = transformer_options.get("patches", {})
        extra = transformer_options.copy()
        qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)),
                               [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        for p in patches.get("attn1_patch", []):
            out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra)
            q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        for p in patches.get("attn1_output_patch", []):
            attn = p(attn, extra)
        if self.yak_mlp:
            mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
        else:
            mlp = self.mlp_act(mlp)
        output = self.linear2(torch.cat((attn, mlp), 2))
        x = x + apply_mod(output, mod.gate, None, modulation_dims)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

    L.DoubleStreamBlock.forward = double_forward
    L.SingleStreamBlock.forward = single_forward
