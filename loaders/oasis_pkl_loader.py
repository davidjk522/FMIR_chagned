"""
utils/getters.py imports `oasis_pkl_loader` from this module for
dataset_name == 'oasisreg', but the file does not exist in the public
FMIR GitHub repo (loaders/ upstream only has acdcreg_loader.py,
abdomenreg_loader.py, abdomenorireg_loader.py, transforms.py).

*** THIS IS A NEW FILE WRITTEN FROM SCRATCH, NOT THE AUTHORS' CODE. ***

It targets the OASIS folder as shipped in this checkout
(./OASIS/imagesTr, labelsTr, masksTr, imagesTs, masksTs,
OASIS_dataset.json) which is the standard Learn2Reg 2021 Task03 "OASIS"
distribution (414 pre-registered/pre-processed volumes at 160x224x192,
36 FreeSurfer-derived label classes 0-35, unpaired).

Pairing/split assumption (please double-check against the paper if exact
reproduction matters): this repo's own abdomenreg_loader.py builds ALL
permutation pairs among train subjects, which works there because it
only has 20 train subjects (380 pairs). OASIS has ~400 train subjects,
so full permutation would give >150,000 pairs/epoch, which nobody
actually trains on. The convention used by essentially every published
OASIS-L2R registration paper (VoxelMorph, TransMorph, LKU-Net, RDP, ...)
instead pairs each subject with its numerical neighbor (i -> i+1):
    - train: subjects 1..395   -> 394 consecutive pairs
    - val:   subjects 396..414 -> 18 consecutive pairs
    - test:  imagesTs (39 images, no labels -> 38 consecutive pairs,
             usable only for qualitative/Dice-free checks; OASIS test
             labels are withheld by the Learn2Reg challenge)
Adjust `n_train_subjects` / `n_val_subjects` below if the paper specifies
a different split.
"""

import os
import glob
import tempfile
import itertools

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class oasis_pkl_loader(Dataset):

    def __init__(self,
            root_dir='./OASIS/',
            split='train',  # train, val or test
            n_train_subjects=395,
            n_val_subjects=19,
            pairing='neighbor',  # 'neighbor' (i, i+1) or 'permutations' (all pairs)
        ):

        self.root_dir = root_dir
        self.split = split

        img_dir = os.path.join(root_dir, 'imagesTr')
        lbl_dir = os.path.join(root_dir, 'labelsTr')

        def fp(d, idx):
            return os.path.join(d, 'OASIS_%04d_0000.nii.gz' % idx)

        if split == 'train':
            idxs = list(range(1, n_train_subjects + 1))
        elif split == 'val':
            idxs = list(range(n_train_subjects + 1, n_train_subjects + n_val_subjects + 1))
        elif split == 'test':
            # OASIS test labels are withheld (Learn2Reg blind test set).
            # imagesTs/masksTs exist but labelsTs does not -> no supervised
            # Dice is possible here, only useful for qualitative checks or
            # for a Learn2Reg-format submission.
            img_dir = os.path.join(root_dir, 'imagesTs')
            lbl_dir = None
            # test subject IDs continue numbering after the training set
            # (observed as OASIS_0415_0000.nii.gz .. OASIS_0453_0000.nii.gz),
            # they do NOT restart at 1.
            test_fps = sorted(glob.glob(os.path.join(img_dir, 'OASIS_*_0000.nii.gz')))
            idxs = sorted(int(os.path.basename(f).split('_')[1]) for f in test_fps)
        else:
            raise ValueError("split must be 'train', 'val' or 'test', got %s" % split)

        self.img_fps = {idx: fp(img_dir, idx) for idx in idxs}
        self.lbl_fps = {idx: fp(lbl_dir, idx) for idx in idxs} if lbl_dir is not None else None

        if pairing == 'neighbor':
            self.sub_idx = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        elif pairing == 'permutations':
            self.sub_idx = list(itertools.permutations(idxs, 2))
        else:
            raise ValueError("pairing must be 'neighbor' or 'permutations'")

        save_fp = os.path.join(root_dir, 'save')
        os.makedirs(save_fp, exist_ok=True)
        self.save_fps = {idx: os.path.join(save_fp, 'subject%04d.npz' % idx) for idx in idxs}

        print('----->>>> %s set has %d subjects' % (split, len(idxs)))
        print('----->>>> %s set has %d pairs (pairing=%s)' % (split, len(self.sub_idx), pairing))

    def __len__(self):
        return len(self.sub_idx)

    def _load_subject(self, idx):
        save_fp = self.save_fps[idx]
        if os.path.exists(save_fp):
            try:
                data = np.load(save_fp)
                return data['img'], data['lbl']
            except Exception as e:
                # A DataLoader worker (num_workers>0) can race another worker
                # that's still mid-write on the same subject's cache the
                # first time it's touched (np.savez() below wasn't atomic),
                # leaving a corrupt/truncated .npz -- e.g. zipfile.BadZipFile.
                # Recompute from the source .nii.gz instead of crashing the
                # whole run; the atomic-rename write below prevents this
                # from happening for any subject going forward.
                print('----->>>> subject # %d cache at %s was corrupt (%s), recomputing' % (idx, save_fp, e))

        img = np.asarray(nib.load(self.img_fps[idx]).get_fdata(), dtype='float32')
        if self.lbl_fps is not None:
            lbl = np.asarray(nib.load(self.lbl_fps[idx]).get_fdata(), dtype='float32')
        else:
            # test split has no ground-truth labels; use an all-zero
            # placeholder so the (img, seg, img, seg) tuple shape stays
            # compatible with the training loop.
            lbl = np.zeros_like(img, dtype='float32')

        img = img[None, ...]
        lbl = lbl[None, ...]
        # Write atomically: savez to a uniquely-named temp file, then
        # os.replace() into place (atomic on POSIX). Without this, two
        # DataLoader worker processes computing the same subject at the
        # same time can both open/write save_fp concurrently and corrupt it
        # -- this is exactly what produced the BadZipFile crash.
        fd, tmp_fp = tempfile.mkstemp(dir=os.path.dirname(save_fp) or '.', suffix='.npz')
        os.close(fd)
        try:
            np.savez(tmp_fp, img=img, lbl=lbl)
            os.replace(tmp_fp, save_fp)
        except BaseException:
            if os.path.exists(tmp_fp):
                os.remove(tmp_fp)
            raise
        print('----->>>> subject # %d saved to %s' % (idx, save_fp))
        return img, lbl

    def __getitem__(self, idx):

        sub_idx1, sub_idx2 = self.sub_idx[idx]

        img1, lbl1 = self._load_subject(sub_idx1)
        img2, lbl2 = self._load_subject(sub_idx2)

        src_img, src_lbl = torch.from_numpy(img1), torch.from_numpy(lbl1)
        tgt_img, tgt_lbl = torch.from_numpy(img2), torch.from_numpy(lbl2)

        return src_img, src_lbl, tgt_img, tgt_lbl, sub_idx1, sub_idx2


if __name__ == '__main__':

    pass
