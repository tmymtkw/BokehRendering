from torch.nn import ZeroPad2d

def padData(x):
    _, _, h, w = x.shape
    t = b = l = r = 0
    is_pad = False
    
    # zero padding
    if h % 8 != 0:
        is_pad = True
        pad_h = 8 - h % 8

        t = pad_h // 2
        b = pad_h - t

    if w % 8 != 0:
        is_pad = True
        pad_w = 8 - w % 8

        l = pad_w // 2
        r = pad_w - l

    if is_pad:
        pad = ZeroPad2d((l, r, t, b))
        x = pad(x)

    return (x, l, r, t, b)

def unpadData(x, l, r, t, b):
    if (l + r + t + b == 0):
        return x
    elif (l + r == 0):
        x = x[:, :, t:-b, :]
    elif (t + b == 0):
        x = x[:, :, :, l:-r]
    else:
        x = x[:, :, t:-b, l:-r]

    return x