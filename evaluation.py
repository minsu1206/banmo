"""
--- Description --- 

Note
    - $seqname : scene name & number which we want to eval. ex) T_swing1, T_samba1, ... 
    - No support about visualization. Only evaluation
    - set proper PYOPENGL_PLATFORM variable

Purpose
    1. Evaluate Chamfer Distance
    2. Evaluate F1-score d=2%

Format
    Input : 
        GT Mesh : obtained from database/$seqname/meshes/mesh_*.obj
        GT Cam. matrix : obtained from database/$seqname/calibration/Camera1.Pmat.cal
        Our Mesh : obtained from v2s_trainer.eval()
        Our Cam. matrix : obtained from v2s_trainer.val()
    Output : Chamfer Distance / F1-score (d=1% | 2% | 5%)

    Note that GT mesh & cam matrix can be obtained from banmo/misc/ama.txt

Reference
    banmo/scripts/visualize/render_vis.py ln355 ~ ln417

"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "osmesa" #opengl seems to only work with TPU
sys.path.insert(0,'third_party')

import glob
from tqdm import tqdm
import logging
import numpy as np
import cv2
import trimesh
import torch
import chamfer3D.dist_chamfer_3D
import fscore
from pytorch3d.ops import iterative_closest_point as icp


DEVICE = 'cuda:0'
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

def get_logger(seqname, outdir):
    logger = logging.getLogger(seqname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    import time
    log_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
    file_handler = logging.FileHandler(f'{outdir}/{seqname}_evaluation.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
    
def obj_to_cam(in_verts, Rmat, Tmat):
    """        From banmo/nnutils/geom_utils.py
    verts: ...,N,3
    Rmat:  ...,3,3
    Tmat:  ...,3 
    """
    verts = in_verts.clone()
    if verts.dim()==2: verts=verts[None]
    verts = verts.view(-1,verts.shape[1],3)
    Rmat = Rmat.view(-1,3,3).permute(0,2,1) # left multiply
    Tmat = Tmat.view(-1,1,3)
    
    verts =  verts.matmul(Rmat) + Tmat 
    verts = verts.reshape(in_verts.shape)
    return verts

def load_gt_cam_matrix(gt_cam_matrix_path):
    pmat = np.loadtxt(gt_cam_matrix_path)
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(pmat)
    Rmat_gt = R
    Tmat_gt = T[:3, 0]/T[-1, 0]
    Tmat_gt = Rmat_gt.dot(-Tmat_gt[..., None])[...,0]
    K = K/K[-1, -1]
    return Rmat_gt, Tmat_gt, K

def load_our_cam_matrix(our_cam_matrix_path):
    """
    Original
        cam_matrix_path = $testdir/$seqname-cam-$seqnumber.txt
                        : path to result of extract.py
    """
    if our_cam_matrix_path[-4:] == '.txt':
        cam = np.loadtxt(our_cam_matrix_path)
    elif our_cam_matrix_path[-4:] == '.npy':
        cam = np.load(our_cam_matrix_path)
    Rmat    = torch.tensor(cam[None, :3, :3]).to(DEVICE)    # rotation matrix
    Tmat    = torch.tensor(cam[None, :3, 3]).to(DEVICE)     # translation vector
    ppoint  = cam[3, 2:]                                    # principal point
    focal   = cam[3, :2]                                    # focal length
    return Rmat, Tmat, ppoint, focal

def measure(verts_gt, verts_ours, bbox_max):
    """
    verts_gt = obj_to_cam(verts_gt, Rmat_gt, Tmat_gt)

    return
        chamfer distance, f1-score d=1%, f1-score d=2%, f1-score d=5%
    """
    # ICP
    fitted_scale = verts_gt[..., -1].median() / verts_ours[..., -1].median()
    verts_ours = verts_ours * fitted_scale
    frts = icp(verts_ours, verts_gt, estimate_scale=False, max_iterations=100)

    verts_ours = ((frts.RTs.s*verts_ours).matmul(frts.RTs.R) + frts.RTs.T[:, None])

    verts_gt = verts_gt.float()      # float64 -> error occurs
    verts_ours = verts_ours.float()    # float64 -> error occurs

    # Chamfer distance & F1-score
    raw_cd, raw_cd_back, _, _ = chamLoss(verts_gt, verts_ours)

    f1, _, _ = fscore.fscore(raw_cd, raw_cd_back, threshold=(bbox_max*0.01)**2)
    f2, _, _ = fscore.fscore(raw_cd, raw_cd_back, threshold=(bbox_max*0.02)**2)
    f5, _, _ = fscore.fscore(raw_cd, raw_cd_back, threshold=(bbox_max*0.05)**2)
    
    raw_cd = np.sqrt(np.asarray(raw_cd.cpu()[0]))
    raw_cd_back = np.sqrt(np.asarray(raw_cd_back.cpu()[0]))
    chamfer_distance = raw_cd.mean() + raw_cd_back.mean()

    return chamfer_distance, f1.cpu().numpy(), f2.cpu().numpy(), f5.cpu().numpy()

def evaluation_one_frame(mesh_gt, mesh_ours, gt_cam_matrix, our_cam_matrix_path):
    # get GT / Ours cam matrix
    Rmat_ours, Tmat_ours, ppoint, focal = load_our_cam_matrix(our_cam_matrix_path)
    Rmat_gt, Tmat_gt, ppoint_gt, focal_gt  = gt_cam_matrix
    if not isinstance(ppoint_gt, type(None)):   # not synthetic data -> change intrinsic params
        ppoint = ppoint_gt
        focal = focal_gt

    # set principal point & focal length as GT's value
    # tensorize cam matrix 
    # Rmat_ours = torch.tensor(Rmat_ours).to(DEVICE)[None]    # (1,3,3)       # FIXME : redundant lines...?
    # Tmat_ours = torch.tensor(Tmat_ours).to(DEVICE)[None]    # (1,3)         # FIXME : redundant lines...?
    Rmat_gt = torch.tensor(Rmat_gt).float().to(DEVICE)[None]    # (1,3,3)
    Tmat_gt = torch.tensor(Tmat_gt).float().to(DEVICE)[None]    # (1,3)
    # tensorize mesh values
    verts_ours = torch.tensor(mesh_ours.vertices[None]).float().to(DEVICE)
    face_ours = torch.tensor(mesh_ours.faces[None]).float().to(DEVICE)
    verts_gt = torch.tensor(mesh_gt.vertices[None]).float().to(DEVICE)
    face_gt = torch.tensor(mesh_gt.faces[None]).float().to(DEVICE)
    Rmat_ours = Rmat_ours.float()
    Tmat_ours = Tmat_ours.float()
    
    bbox_max = float((verts_gt.max(1)[0] - verts_gt.min(1)[0]).max().cpu())
    # apply RT into vertices
    verts_ours = obj_to_cam(verts_ours, Rmat_ours, Tmat_ours)
    verts_gt = obj_to_cam(verts_gt, Rmat_gt, Tmat_gt)

    # calculate distance & score
    chamfer_distance, f1, f2, f5 = measure(verts_gt=verts_gt, verts_ours=verts_ours, bbox_max=bbox_max)
    return chamfer_distance, f1, f2, f5

def evaluate(gt_mesh_dir, our_dir, gt_cam_matrix_path, seqname, cam_ext, prefix, zfill, logger):

    mesh_prefix = '-mesh-' if prefix == '' else prefix
    cam_prefix = '-cam-' if prefix == '' else prefix
    # prepared data : gt cam
    if 'a-hands' in seqname or 'a-eagle' in seqname:
        # when we use Synthetic data (ex. a-eagle, a-hands)
        Rmat_gt = np.eye(3)
        Tmat_gt = np.asarray([0, 0, 0])
        ppoint_gt = None
        focal_gt = None
    else:
        Rmat_gt, Tmat_gt, K  = load_gt_cam_matrix(gt_cam_matrix_path)
        ppoint_gt = np.array([K[0, 2], K[1, 2]])
        focal_gt  = np.array([K[0, 0], K[1, 1]])

    gt_cam_matrix = [Rmat_gt, Tmat_gt, ppoint_gt, focal_gt]

    # prepare data : gt_mesh / our_mesh / our_camera
    gt_meshes = []
    our_meshes = []
    our_cam_paths = []
    logger.info('Load GT Meshes')

    for gt_mesh_path in tqdm(sorted(glob.glob(f'{gt_mesh_dir}/*.obj'))):
        gt_meshes.append(trimesh.load(gt_mesh_path, process=False))

    logger.info('Load Our Meshes / Cam matrix')
    for idx in tqdm(range(len(gt_meshes))):
        our_mesh_path = f'{our_dir}/{seqname}{mesh_prefix}{str(idx).zfill(zfill)}.obj'
        # our_cam_path = our_mesh_path.replace(mesh_prefix, cam_prefix).replace('.obj', '.txt')
        our_cam_path = f'{our_dir}/{seqname}{cam_prefix}{str(idx).zfill(zfill)}{cam_ext}'
        try: 
            our_meshes.append(trimesh.load(our_mesh_path, process=False))
            our_cam_paths.append(our_cam_path)
        except:     # FIXME : different length ...? 
            print(our_mesh_path, our_cam_path)
            break

    gt_meshes = gt_meshes[:len(our_meshes)]

    # evaluation loop
    all_chamfer_distance = []
    all_f1_score_d1 = []
    all_f1_score_d2 = []
    all_f1_score_d5 = []

    logger.info("Start Evaluation ...")
    
    for i, (gt_mesh, our_mesh, our_cam_path) in enumerate(zip(gt_meshes, our_meshes, our_cam_paths)):
        chamfer_distance, f1_score_d1, f1_score_d2, f1_score_d5 = evaluation_one_frame(gt_mesh, our_mesh, gt_cam_matrix, our_cam_path)
        logger.info(f'Frame-{str(i).zfill(4)} | CD : {round(float(chamfer_distance), 3)} | F1-score d=2% : {round(float(f1_score_d2), 3)}')
        all_chamfer_distance.append(chamfer_distance * 100)
        all_f1_score_d1.append(f1_score_d1)
        all_f1_score_d2.append(f1_score_d2)
        all_f1_score_d5.append(f1_score_d5)

    # statistics
    avg_chamfer_distance = np.mean(all_chamfer_distance)
    max_chamfer_distance = np.max(all_chamfer_distance)
    min_chamfer_distance = np.min(all_chamfer_distance)
    std_chamfer_distance = np.std(all_chamfer_distance)

    avg_f1_score_d1 = np.mean(all_f1_score_d1) * 100 
    avg_f1_score_d2 = np.mean(all_f1_score_d2) * 100
    avg_f1_score_d5 = np.mean(all_f1_score_d5) * 100 

    max_f1_score_d1 = np.max(all_f1_score_d1) * 100
    max_f1_score_d2 = np.max(all_f1_score_d2) * 100
    max_f1_score_d5 = np.max(all_f1_score_d5) * 100
    
    min_f1_score_d1 = np.min(all_f1_score_d1) * 100
    min_f1_score_d2 = np.min(all_f1_score_d2) * 100
    min_f1_score_d5 = np.min(all_f1_score_d5) * 100

    std_f1_score_d1 = np.std(all_f1_score_d1) * 100
    std_f1_score_d2 = np.std(all_f1_score_d2) * 100
    std_f1_score_d5 = np.std(all_f1_score_d5) * 100
    
    # records
    logger.info('Chamfer Distance')
    logger.info(f'AVG : {avg_chamfer_distance.round(3)} | STD : {std_chamfer_distance.round(3)}')
    logger.info(f'MAX : {max_chamfer_distance.round(3)} | MIN : {min_chamfer_distance.round(3)}')

    logger.info('F1-score d=1%')
    logger.info(f'AVG : {avg_f1_score_d1.round(3)} | STD : {std_f1_score_d1.round(3)}')
    logger.info(f'MAX : {max_f1_score_d1.round(3)} | MIN : {min_f1_score_d1.round(3)}')

    logger.info('F1-score d=2%')
    logger.info(f'AVG : {avg_f1_score_d2.round(3)} | STD : {std_f1_score_d2.round(3)}')
    logger.info(f'MAX : {max_f1_score_d2.round(3)} | MIN : {min_f1_score_d2.round(3)}')

    logger.info('F1-score d=5%')
    logger.info(f'AVG : {avg_f1_score_d5.round(3)} | STD : {std_f1_score_d5.round(3)}')
    logger.info(f'MAX : {max_f1_score_d5.round(3)} | MIN : {min_f1_score_d5.round(3)}')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqname', type=str, default='T_swing1')
    parser.add_argument('--gt_mesh_dir', type=str, default='database/T_swing/meshes')
    parser.add_argument('--gt_cam_path', type=str, default='database/T_swing/calibration/Camera1.Pmat.cal')
    parser.add_argument('--our_dir', type=str, default='logdir/T_swing_meao_warmup_ver1.2.8')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--type_prefix', type=str, default='')
    parser.add_argument('--mesh_ext', type=str, default='.obj')
    parser.add_argument('--cam_ext', type=str, default='.txt')
    parser.add_argument('--zfill', type=int, default=5)
    args = parser.parse_args()

    if args.device != DEVICE:
        DEVICE = args.device

    log_dir = args.our_dir.replace('extract', 'eval')
    os.makedirs(log_dir, exist_ok=True)

    if args.seqname == 'T_samba1':
        gt_mesh_dir = args.gt_mesh_dir.replace('T_swing', 'T_samba')
        gt_cam_path = args.gt_cam_path.replace('T_swing', 'T_samba')
    elif args.seqname == 'a-eagle-1':
        gt_mesh_dir = 'database/DAVIS_syn/Meshes/Full-Resolution/a-eagle-1'
        gt_cam_path = 'database/DAVIS_syn/Cameras/Full-Resolution/a-eagle-1'
    elif args.seqname == 'a-hands-1':
        gt_mesh_dir = 'database/DAVIS_hands/Meshes/Full-Resolution/a-hands-1'
        gt_cam_path = 'database/DAVIS_hands/Cameras/Full-Resolution/a-hands-1'
    else:
        gt_mesh_dir = args.gt_mesh_dir
        gt_cam_path = args.gt_cam_path
    
    print(f"GT path : {gt_mesh_dir} {gt_cam_path}")
    logger = get_logger(args.seqname, log_dir)

    evaluate(gt_mesh_dir, args.our_dir, gt_cam_path, args.seqname, args.cam_ext, args.type_prefix, args.zfill, logger)







    



