import numpy as np

def im2col(x, kernel_size, strides, pad, out_shape): 
	#x: [N, H ,W, C]
	x = x.transpose(0, 3, 1, 2) # [N, C, H, W]
	
	N, C, H, W = x.shape
	FH, FW = kernel_size
	SH, SW = strides
	out_H, out_W = out_shape

	H_up_pad = pad[0]//2
	H_down_pad = pad[0] - H_up_pad
	W_left_pad = pad[1]//2
	W_right_pad = pad[1] - W_left_pad

	#pad
	pad_x = np.pad(x, [(0,0), (0,0), # [N, C, H+pad[0], W+pad[1]]
			(H_up_pad, H_down_pad), (W_left_pad, W_right_pad)], 'constant')

	col = np.zeros([N, out_H*out_W, FH*FW*C], np.float32)
	count = 0
	
	for h in range(out_H): 
		for w in range(out_W): 
			col[:, count, :] = pad_x[:, :, 
						(SH*h) : (SH*h) + FH,
						(SW*w) : (SW*w) + FW].reshape(N, FH*FW*C)
			count+=1
	
	col = col.reshape(N*out_H*out_W, FH*FW*C) 
	return col


def col2im(col, x_shape, kernel_size, strides, pad, out_shape):
	#col: [N*out_H,out*W, FH*FW*C]
	N, _, _, C = x_shape # [N, H, W, C]
	pad_H, pad_W = x_shape[1:3] + pad

	FH, FW = kernel_size
	SH, SW = strides
	out_H, out_W = out_shape

	H_up_pad = pad[0]//2
	H_down_pad = pad[0] - H_up_pad
	W_left_pad = pad[1]//2
	W_right_pad = pad[1] - W_left_pad

	col = col.reshape(N, out_H*out_W, FH*FW*C)
	
	x = np.zeros([N, C, pad_H, pad_W], np.float32)
	count = 0

	for h in range(out_H):
		for w in range(out_W):
			x[:, :,
				(SH*h) : (SH*h) + FH,
				(SW*w) : (SW*w) + FW] = col[:, count, :].reshape(N, C, FH, FW)
			count+=1

	#unpad
	x = x[:, :,		  # [N, C, H, W]
			H_up_pad : pad_H - H_down_pad, 
			W_left_pad : pad_W - W_right_pad]

	x = x.transpose(0,2,3,1) # [N, H, W, C] == x_shape
	return x
