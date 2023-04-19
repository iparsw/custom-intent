iparsw custom up sampling loss function

it is defined as :
```python
def IPCUSL_loss(y_true, y_pred):
    return (charbonnier_loss(y_true, y_pred)) * (SSIM_loss(y_true, y_pred)) * (
            PSNR_loss(y_true, y_pred)) * (TV_loss(y_true, y_pred))
```
note that this might change because im currently experimenting with different configurations