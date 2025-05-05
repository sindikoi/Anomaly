import torch
import os
import numpy as np
from options import parse_args
from config import Config
from dataset_loader import UCF_crime
from model import WSAD
from dataset_loader import data
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def valid(net, config, test_loader, model_file=None):
    print("[DEBUG] Starting validation...")

    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file, map_location='cuda:0'))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None

        ucf_pdict = {
            "Abuse": {}, "Arrest": {}, "Arson": {}, "Assault": {}, "Burglary": {},
            "Explosion": {}, "Fighting": {}, "RoadAccidents": {}, "Robbery": {},
            "Shooting": {}, "Shoplifting": {}, "Stealing": {}, "Vandalism": {}, "Normal": {}
        }
        ucf_gdict = {k: {} for k in ucf_pdict}

        cls_label = []
        cls_pre = []
        temp_predict = []
        count = 0

        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            _name = _name[0]
            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict_tensor = res["frame"]  # shape: [segments, 1]
            temp_predict.append(a_predict_tensor)

            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))

                # Safe averaging even with varying sizes
                a_list = [t.mean().item() for t in temp_predict]  # כל טנסור → מספר בודד
                a_predict = np.array([np.mean(a_list)])  # הפוך למערך בגודל 1

                print(f"[DEBUG] a_predict shape: {a_predict.shape}")

                pl = len(a_predict) * 16

                try:
                    if "Normal" in _name:
                        ucf_pdict["Normal"][_name] = np.repeat(a_predict, 16)
                        ucf_gdict["Normal"][_name] = frame_gt[count:count + pl]
                    else:
                        ucf_pdict[_name[:-3]][_name] = np.repeat(a_predict, 16)
                        ucf_gdict[_name[:-3]][_name] = frame_gt[count:count + pl]
                except KeyError as e:
                    print(f"[WARNING] {e} not found in dictionary – skipping")

                count += pl
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

                temp_predict = []

        # 🧠 match gt length to prediction length
        frame_gt = frame_gt[:len(frame_predict)]

        # Save results
        np.save('frame_label/ucf_pre.npy', frame_predict)
        np.save('frame_label/ucf_pre_dict.npy', ucf_pdict)
        np.save('frame_label/ucf_gt_dict.npy', ucf_gdict)

        # Metrics
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
        accuracy = np.sum(np.array(cls_label) == np.array(cls_pre)) / len(cls_pre)

        print(f"AUC: {auc_score:.4f}")
        print(f"AP Score: {ap_score:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    print("[DEBUG] Entered main")
    args = parse_args()
    config = Config(args)

    print("[DEBUG] Creating model")
    net = WSAD(input_size=config.len_feature, flag="Test", a_nums=60, n_nums=60)
    net = net.cuda()

    print("[DEBUG] Creating test loader")
    test_loader = data.DataLoader(
        UCF_crime(
            root_dir=config.root_dir,
            mode='Test',
            modal=config.modal,
            num_segments=config.num_segments,
            len_feature=config.len_feature
        ),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=None
    )

    print("[DEBUG] Calling valid()")
    valid(net, config, test_loader, model_file=os.path.join(args.model_path, args.model_file))
