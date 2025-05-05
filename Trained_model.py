import torch

# טען את המודל המאומן
model_path = "models/ucf_trans_2022.pkl"  # או xd_trans_2022.pkl
model = torch.load(model_path)

# אם יש צורך בטעינת המודל למכשיר CUDA
model = model.cuda()


