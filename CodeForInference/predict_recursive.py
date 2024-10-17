from labelSys.utils.labelReaderV2 import recursivelyFindLabelDir
from UNetPPTMJ.trainer.trainerTMJ import TrainerTMJ
from UNetPPTMJ.dataloading.rawDataPreparation import getImgsMsksAndSpacing
from monsoonToolBox.misc import divideChunks
import torch
import numpy as np
import tqdm
import argparse, os, glob, hashlib
from typing import List


DEVICE = "cpu"

def getImagesAndMasks(pth: List[str], path_base = ""):
    images = []
    masks = []
    for p in pth:
        im, msk, _ = getImgsMsksAndSpacing(p)
        images.append(im)
        masks.append(msk)
    if path_base != "":
        pth = [os.path.relpath(p, path_base) for p in pth]
    return images, masks, pth

def _predict(images, trainers: TrainerTMJ, batch_size = 4):
    patient_masks_pred = None
    patient_masks_preds = []
    print(trainers)
    for trainer in trainers:
        imgs_batched = divideChunks(images, batch_size)
        trainer_pred = []
        for img_batch in imgs_batched:
            img_tensor = torch.tensor(img_batch, dtype=torch.float32).to(DEVICE)
            img_tensor.unsqueeze_(1)
            _pred = trainer.model(img_tensor).detach()
            trainer_pred.append(_pred)
        trainer_pred = torch.cat(trainer_pred, dim=0)
        patient_masks_preds.append(trainer_pred.cpu().numpy())
    patient_masks_preds = torch.tensor(patient_masks_preds, device=DEVICE)
    patient_masks_pred = patient_masks_preds.mean(dim=0)
    del patient_masks_preds
    patient_masks_pred = patient_masks_pred.argmax(dim=1).cpu().numpy()
    return patient_masks_pred

def predictOneLabelDir(label_dir: str, path_base: str, trainers: TrainerTMJ, batch_size = 4):
    images, masks, paths = getImagesAndMasks([label_dir], path_base)
    # images, masks: [1, n_slices, 256, 256]
    # paths: [1]

    imgs = images[0]  # (n_slices, 256, 256)
    msks = masks[0]   # (n_slices, 256, 256)
    pth = paths[0]    # str

    pred_msks = _predict(imgs, trainers, batch_size)
    return {
        "images": np.array(imgs),
        "masks": np.array(msks),
        "pred_masks": pred_msks,
        "path": pth
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, 
        help = "Top level directory of the data, will recursively find all the label directories")
    parser.add_argument("-f", type=str, default=[], nargs="*", dest="save_dirs", help = "model save directoies")
    parser.add_argument("-b", action="store", default=4, dest="batch_size", help = "batch size")
    parser.add_argument(
        "-o", "--output_dir", default = "./predict_output", 
        help="output directory, will save along side with the input data path in txt files",
        )
    args = parser.parse_args()
    
    save_dirs = []
    for save_dir in args.save_dirs:
        save_dirs.extend(glob.glob(save_dir))

    batch_size = int(args.batch_size)

    trainers = [TrainerTMJ() for _ in range(len(save_dirs))]
    for i in range(len(save_dirs)):
        trainers[i].model = trainers[i].model.to(DEVICE)
        trainers[i].loadModel(save_dir=save_dirs[i], mode="best", device=DEVICE)
        trainers[i].model.eval()
    print("Loaded {} trainers".format(len(trainers)))

    all_label_dir = recursivelyFindLabelDir(args.data_path)
    print("Found {} label directories".format(len(all_label_dir)))

    if input("Continue? (y/n): ") != "y":
        exit(0)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for label_dir in tqdm.tqdm(all_label_dir):

        with torch.no_grad():
            output = predictOneLabelDir(
                label_dir, 
                path_base=args.data_path, 
                trainers=trainers, 
                batch_size=batch_size
                )
        
        label_dir_rel = os.path.relpath(label_dir, args.data_path)
        __this_output_dir_name = hashlib.md5(label_dir_rel.encode()).hexdigest()
        this_output_dir = os.path.join(args.output_dir, __this_output_dir_name)
        if not os.path.exists(this_output_dir):
            os.mkdir(this_output_dir)
        
        np.savez(
            os.path.join(this_output_dir, "result.npz"),
            images=output["images"],
            masks=output["masks"],
            pred_masks=output["pred_masks"],
            paths=output["path"],
            )
        with open(os.path.join(this_output_dir, "source.txt"), "w") as fp:
            fp.write(label_dir_rel)
        