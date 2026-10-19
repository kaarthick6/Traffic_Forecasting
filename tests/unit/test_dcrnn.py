import pytest
torch=pytest.importorskip("torch")
from app.ml.dcrnn import DCRNN, masked_mae
from app.ml.graph import diffusion_supports

def test_dcrnn_forward_shapes_and_loss():
    model=DCRNN(num_nodes=4,hidden_channels=8,diffusion_order=2,encoder_layers=2,decoder_layers=2,horizon=3)
    f,b=diffusion_supports([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]],2)
    x=torch.randn(2,12,4,1); y=model(x,(torch.tensor(f),torch.tensor(b)))
    assert y.shape==(2,3,4,1)
    loss=masked_mae(y,torch.zeros_like(y)); assert torch.isfinite(loss)

def test_checkpoint_save_load_and_prediction_consistency(tmp_path):
    model=DCRNN(num_nodes=2,hidden_channels=4,diffusion_order=1,encoder_layers=1,decoder_layers=1,horizon=2)
    supports=tuple(torch.tensor(x) for x in diffusion_supports([[0,1],[1,0]],1))
    x=torch.randn(1,4,2,1); expected=model(x,supports)
    checkpoint=tmp_path/"dcrnn.pt"; torch.save(model.state_dict(),checkpoint)
    restored=DCRNN(num_nodes=2,hidden_channels=4,diffusion_order=1,encoder_layers=1,decoder_layers=1,horizon=2)
    restored.load_state_dict(torch.load(checkpoint,weights_only=True))
    assert torch.allclose(expected,restored(x,supports))

