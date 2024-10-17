import argparse, os, pickle

import torch
import numpy as np
from progress.bar import Bar
from UNetPPTMJ.trainer.trainerTMJ import TrainerTMJ
from UNetPPTMJ.config import TEMP_DIR
from UNetPPTMJ.dataloading.rawDataPreparation import getImgsMsksAndSpacing
from monsoonToolBox.misc import divideChunks
from typing import List

def getImagesAndMasks(pth: List[str]):
    images = []
    masks = []
    for p in pth:
        im, msk, _ = getImgsMsksAndSpacing(p)
        images.append(im)
        masks.append(msk)
    return images, masks, pth

def main():
    default_result_path = os.path.join(TEMP_DIR, "result-upp.pkl")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type = str, default=[], nargs="+", help = "labeled directories to be predicted")
    parser.add_argument("-f", type=str, default=[], nargs="*", dest="save_dirs", help = "model save directories")
    parser.add_argument("-b", type=int, default=4, dest="batch_size", help = "batch size")
    parser.add_argument("-o", "--output", default = default_result_path, help="output file, will be saved in python pickle file")
    args = parser.parse_args()

    save_dirs = args.save_dirs
    batch_size = args.batch_size

    images, masks, paths = getImagesAndMasks(args.data)
    print(f"{len(images[0])} images loaded")

    device = "cpu"
    trainers = [TrainerTMJ() for _ in range(len(save_dirs))]
    for i in range(len(save_dirs)):
        trainers[i].model = trainers[i].model.to(device)
        trainers[i].loadModel(save_dir=save_dirs[i], mode="best", device=device)
        trainers[i].model.eval()

    pred_masks = []
    progress = Bar(max=len(images))
    for imgs, msks in zip(images, masks):
        print(len(imgs), len(msks))
        imgs_batched = list(divideChunks(imgs, batch_size))
        patient_images = imgs
        patient_masks = msks
        patient_masks_pred = None
        patient_masks_preds = []
        for trainer in trainers:
            trainer_pred = []
            for i in range(len(imgs_batched)):
                img_batch = imgs_batched[i]
                img_tensor = torch.tensor(img_batch, dtype=torch.float32).to(device)
                img_tensor.unsqueeze_(1)
                _pred = trainer.model(img_tensor).detach()
                trainer_pred.append(_pred)
            trainer_pred = torch.cat(trainer_pred, dim=0)
            patient_masks_preds.append(trainer_pred.cpu().numpy())
        patient_masks_preds = torch.tensor(patient_masks_preds, device=device)
        patient_masks_pred = patient_masks_preds.mean(dim=0)
        del patient_masks_preds
        patient_masks_pred = patient_masks_pred.argmax(dim=1).cpu().numpy()

        pred_masks.append(patient_masks_pred)
        progress.next()
    # import pdb; pdb.set_trace()
    # np.savez(result_path, imgs=images, masks=pred_masks, labels=masks, dtype=object)
    with open(args.output, 'wb') as fp:
        pickle.dump(
            {
                "imgs": images,
                "masks": pred_masks,
                "labels": masks,
                "paths": paths
            },
            fp
        )
    print("\nDone!, result saved to: ", args.output)


if __name__ == "__main__":
    main()
