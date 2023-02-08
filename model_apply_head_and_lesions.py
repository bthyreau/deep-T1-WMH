import torch
import nibabel
import numpy as np
import os, sys, time
import scipy.ndimage
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import inv
import resource

if len(sys.argv[1:]) == 0:
    print("Need to pass one or more T1 image filename as argument")
    sys.exit(1)


if 0: # change here to activate thread limitation
    torch.set_num_threads(4)

print("Using %d CPU threads" % torch.get_num_threads())


class HeadModel(nn.Module):
    def __init__(self):
        super(HeadModel, self).__init__()
        self.conv0a = nn.Conv3d(1, 8, 3, padding=1)
        self.conv0b = nn.Conv3d(8, 8, 3, padding=1)
        self.bn0a = nn.BatchNorm3d(8)

        self.ma1 = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(8, 16, 3, padding=1)
        self.conv1b = nn.Conv3d(16, 24, 3, padding=1)
        self.bn1a = nn.BatchNorm3d(24)

        self.ma2 = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(24, 24, 3, padding=1)
        self.conv2b = nn.Conv3d(24, 32, 3, padding=1)
        self.bn2a = nn.BatchNorm3d(32)

        self.ma3 = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(32, 48, 3, padding=1)
        self.conv3b = nn.Conv3d(48, 48, 3, padding=1)
        self.bn3a = nn.BatchNorm3d(48)


        self.conv2u = nn.Conv3d(48, 24, 3, padding=1)
        self.conv2v = nn.Conv3d(24+32, 24, 3, padding=1)
        self.bn2u = nn.BatchNorm3d(24)


        self.conv1u = nn.Conv3d(24, 24, 3, padding=1)
        self.conv1v = nn.Conv3d(24+24, 24, 3, padding=1)
        self.bn1u = nn.BatchNorm3d(24)


        self.conv0u = nn.Conv3d(24, 16, 3, padding=1)
        self.conv0v = nn.Conv3d(16+8, 8, 3, padding=1)
        self.bn0u = nn.BatchNorm3d(8)

        self.conv1x = nn.Conv3d(8, 4, 1, padding=0)

    def forward(self, x):
        x = F.elu(self.conv0a(x))
        self.li0 = x = F.elu(self.bn0a(self.conv0b(x)))

        x = self.ma1(x)
        x = F.elu(self.conv1a(x))
        self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))

        x = self.ma2(x)
        x = F.elu(self.conv2a(x))
        self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))

        x = self.ma3(x)
        x = F.elu(self.conv3a(x))
        self.li3 = x = F.elu(self.bn3a(self.conv3b(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.conv2u(x))
        x = torch.cat([x, self.li2], 1)
        x = F.elu(self.bn2u(self.conv2v(x)))

        self.lo1 = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.conv1u(x))
        x = torch.cat([x, self.li1], 1)
        x = F.elu(self.bn1u(self.conv1v(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        self.la1 = x

        x = F.elu(self.conv0u(x))
        x = torch.cat([x, self.li0], 1)
        x = F.elu(self.bn0u(self.conv0v(x)))

        self.out = x = self.conv1x(x)
        x = torch.sigmoid(x)
        return x




class ModelAff(nn.Module):
    def __init__(self):
        super(ModelAff, self).__init__()
        self.convaff1 = nn.Conv3d(2, 16, 3, padding=1)
        self.maaff1 = nn.MaxPool3d(2)
        self.convaff2 = nn.Conv3d(16, 16, 3, padding=1)
        self.bnaff2 = nn.LayerNorm([32, 32, 32])

        self.maaff2 = nn.MaxPool3d(2)
        self.convaff3 = nn.Conv3d(16, 32, 3, padding=1)
        self.bnaff3 = nn.LayerNorm([16, 16, 16])

        self.maaff3 = nn.MaxPool3d(2)
        self.convaff4 = nn.Conv3d(32, 64, 3, padding=1)
        self.maaff4 = nn.MaxPool3d(2)
        self.bnaff4 = nn.LayerNorm([8, 8, 8])
        self.convaff5 = nn.Conv3d(64, 128, 1, padding=0)
        self.convaff6 = nn.Conv3d(128, 12, 4, padding=0)

        gsx, gsy, gsz = 64, 64, 64
        gx, gy, gz = np.linspace(-1, 1, gsx), np.linspace(-1, 1, gsy), np.linspace(-1,1, gsz)
        grid = np.meshgrid(gx, gy, gz) # Y, X, Z
        grid = np.stack([grid[2], grid[1], grid[0], np.ones_like(grid[0])], axis=3)
        netgrid = np.swapaxes(grid, 0, 1)[...,[2,1,0,3]]
        
        self.register_buffer('grid', torch.tensor(netgrid.astype("float32"), requires_grad = False))
        self.register_buffer('diagA', torch.eye(4, dtype=torch.float32))

    def forward(self, outc1):
        x = outc1
        x = F.relu(self.convaff1(x))
        x = self.maaff1(x)
        x = F.relu(self.bnaff2(self.convaff2(x)))
        x = self.maaff2(x)
        x = F.relu(self.bnaff3(self.convaff3(x)))
        x = self.maaff3(x)
        x = F.relu(self.bnaff4(self.convaff4(x)))
        x = self.maaff4(x)
        x = F.relu(self.convaff5(x))
        x = self.convaff6(x)

        x = x.view(-1, 3, 4)
        x = torch.cat([x, x[:,0:1] * 0], dim=1)
        self.tA = torch.transpose(x + self.diagA, 1, 2)

        wgrid = self.grid @ self.tA[:,None,None]
        gout = F.grid_sample(outc1, wgrid[...,[2,1,0]], align_corners=True)
        return gout, self.tA

    def resample_other(self, other):
        with torch.no_grad():
            wgrid = self.grid @ self.tA[:,None,None]
            gout = F.grid_sample(other, wgrid[...,[2,1,0]], align_corners=True)
            return gout



def bbox_world(affine, shape):
    s = shape[0]-1, shape[1]-1, shape[2]-1
    bbox = [[0,0,0], [s[0],0,0], [0,s[1],0], [0,0,s[2]], [s[0],s[1],0], [s[0],0,s[2]], [0,s[1],s[2]], [s[0],s[1],s[2]]]
    w = affine @ np.column_stack([bbox, [1]*8]).T
    return w.T

bbox_one = np.array([[-1,-1,-1,1], [1, -1, -1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [1,1,1,1]])

affine64_mni = \
np.array([[  -2.85714293,   -0.        ,    0.        ,   90.        ],
          [  -0.        ,    3.42857146,   -0.        , -126.        ],
          [   0.        ,    0.        ,    2.85714293,  -72.        ],
          [   0.        ,    0.        ,    0.        ,    1.        ]])


scriptpath = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cpu")
net = HeadModel()
net.to(device)
net.load_state_dict(torch.load(scriptpath + "/torchparams/params_head_00075_00000.pt", map_location=device))
net.eval()

netAff = ModelAff()
netAff.load_state_dict(torch.load(scriptpath + "/torchparams/paramsaffineta_00079_00000.pt", map_location=device), strict=False)
netAff.to(device)
netAff.eval()

class Cap5eModel(nn.Module):
    def __init__(self):
        super(Cap5eModel, self).__init__()

        self.block0 = nn.Sequential( nn.Conv3d(1, 24, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
                                     nn.Conv3d(24, 64, 3, padding=1), nn.ReLU(inplace=True) )

        self.ma1 = nn.MaxPool3d(2, return_indices=True)
        self.block1 = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1),  nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
                                     nn.Conv3d(64, 64, 3, padding=1),  nn.ReLU(inplace=True),)

        self.ma2 = nn.MaxPool3d(2, return_indices=True)
        self.block2 = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1),  nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
                                     nn.Conv3d(64, 64, 3, padding=1),  nn.ReLU(inplace=True),)

        self.ma3 = nn.MaxPool3d((2,2,2), return_indices=True)
        self.block3 = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1),  nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
                                     nn.Conv3d(64, 64, 3, padding=1),  nn.ReLU(inplace=True),)

        self.ma4 = nn.MaxPool3d((2,2,1), return_indices=True)
        self.block4 = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1),  nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
                                     nn.Conv3d(64, 64, 3, padding=1),  nn.ReLU(inplace=True),
                                   )

        self.ma5 = nn.MaxPool3d((2,2,1), return_indices=True)
        self.block5 = nn.Sequential( nn.Conv3d(64, 128, 3, padding=1),  nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
                                     nn.Conv3d(128, 64, 3, padding=1),  nn.ReLU(inplace=True),
                                   )

        self.blockUp4_a = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=True),)
        self.blockUp4_b = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(inplace=True) )

        self.blockUp3_a = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=False),)
        self.blockUp3_b = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(inplace=True) )


        self.blockUp2_a = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=False),)
        self.blockUp2_b = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(inplace=True) )

        self.blockUp1_a = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=False), )
        self.blockUp1_b = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(inplace=True) )

        self.blockUp0_a = nn.Sequential( nn.Conv3d(64, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=False), )
        self.blockUp0_b1 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(inplace=True) )
        self.blockUp0_b2 = nn.Sequential( nn.Conv3d(64, 24, 3, padding=1), nn.ReLU(inplace=True) )

        self.conv1x = nn.Conv3d(24, 2, 1, padding=0)
        #self.conv1x = nn.Conv3d(8, 3, 1, padding=1)

        self.uma = nn.MaxUnpool3d(2)
        self.uma4 = nn.MaxUnpool3d((2,2,1))

    def forward(self, x):
        self.li0 = x = self.block0(x)

        x, ind1 = self.ma1(x)
        self.li1 = x = self.block1(x)

        x, ind2 = self.ma2(x)
        self.li2 = x = self.block2(x)
 
        x, ind3 = self.ma3(x)
        self.li3 = x = self.block3(x)
 
        x, ind4 = self.ma4(x)
        self.li4 = x = self.block4(x)
 
        x, ind5 = self.ma5(x)
        x = self.block5(x)

        x = self.uma4(x, ind5, output_size = self.li4.shape)
        x = self.blockUp4_a(x)
        x = self.blockUp4_b(x)
 
        x = self.uma4(x, ind4, output_size = self.li3.shape)
        self.lo3 = x = self.blockUp3_a(x)
        x += self.li3
        self.lo3a = x = self.blockUp3_b(x)


        x = self.uma(x, ind3, output_size = self.li2.shape)
        self.lo2 = x = self.blockUp2_a(x)
        x += self.li2
        self.lo2a = x = self.blockUp2_b(x)

        #x = F.interpolate(x, self.li1.shape[-3:])
        x = self.uma(x, ind2, output_size = self.li1.shape)
        self.lo1 = x = self.blockUp1_a(x)
        x += self.li1
        self.lo1a = x = self.blockUp1_b(x)

        #x = F.interpolate(x, self.li0.shape[-3:])
        x = self.uma(x, ind1, output_size = self.li0.shape)
        x = self.blockUp0_a(x)
        x += self.li0
        x = self.blockUp0_b1(x)
        x = self.blockUp0_b2(x)

        x = self.conv1x(x)
        return torch.sigmoid(x)
ac = Cap5eModel()
#ac.load_state_dict(torch.load(scriptpath + "/torchparams/params_finalwide_train4096.pt", map_location=device), strict=False)
ac.load_state_dict(torch.load(scriptpath + "/torchparams/params_finalwide_train7500.pt", map_location=device), strict=False)
ac.to(device)
ac.eval()


class Cap6Model(nn.Module):
    def __init__(self):
        super(Cap6Model, self).__init__()

        self.block0 = nn.Sequential( nn.Conv3d(1, 12, 3, padding=1), nn.ReLU(inplace=True) )

        self.ma1 = nn.MaxPool3d(2, return_indices=False)
        self.block1 = nn.Sequential( nn.Conv3d(12, 16, 3, padding=1),  nn.ReLU(inplace=True),
                                     nn.Conv3d(16, 16, 1, padding=0),  nn.ReLU(inplace=True),)

        self.ma2 = nn.MaxPool3d(2, return_indices=True)
        self.block2 = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1),  nn.ReLU(inplace=True),
                                     nn.Conv3d(16, 16, 1, padding=0),  nn.ReLU(inplace=True),)

        self.ma3 = nn.MaxPool3d(2, return_indices=True)
        self.block3 = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1),  nn.ReLU(inplace=True),
                                     nn.Conv3d(16, 16, 1, padding=0),  nn.ReLU(inplace=True),)

        self.ma4 = nn.MaxPool3d(2, return_indices=True)
        self.block4 = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1),  nn.ReLU(inplace=True),
                                     nn.Conv3d(16, 16, 1, padding=0),  nn.ReLU(inplace=True),
                                   )

        self.ma5 = nn.MaxPool3d((2,2,1), return_indices=True)
        self.block5 = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1),  nn.ReLU(inplace=True),
                                     nn.Conv3d(16, 16, 1, padding=0),  nn.InstanceNorm3d(16), nn.ReLU(inplace=True) )


        self.blockUp4_a = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True),)
        self.blockUp4_b = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(inplace=True) )

        self.blockUp3_a = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True),)
        self.blockUp3_b = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(inplace=True) )


        self.blockUp2_a = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True),)
        self.blockUp2_b = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(inplace=True) )

        self.blockUp1_a = nn.Sequential( nn.Conv3d(16, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True), )
        self.blockUp1_b = nn.Sequential( nn.Conv3d(16, 12, 3, padding=1), nn.ReLU(inplace=True) )

        self.blockUp0_a = nn.Sequential( nn.Conv3d(12, 12, 3, padding=1), nn.ReLU(inplace=True), )
        self.blockUp0_b1 = nn.Sequential( nn.Conv3d(12, 12, 1, padding=1), nn.ReLU(inplace=True) )
        self.blockUp0_b2 = nn.Sequential( nn.Conv3d(12, 8, 3, padding=0), nn.ReLU(inplace=True) )

        ##self.conv1x = nn.Conv3d(12, 1, 1, padding=0)
        self.conv1x = nn.Conv3d(8, 4, 1, padding=0)
        ##self.conv1x = nn.Conv3d(6, 1, 1, padding=0)

        self.uma = nn.MaxUnpool3d(2)
        self.uma5 = nn.MaxUnpool3d((2,2,1))

    def forward(self, x):
        self.li0 = x = self.block0(x)

        sh1 = x.shape
        x = self.ma1(x)
        self.li1 = x = self.block1(x)

        sh2 = x.shape
        x, ind2 = self.ma2(x)
        self.li2 = x = self.block2(x)
 
        sh3 = x.shape
        x, ind3 = self.ma3(x)
        self.li3 = x = self.block3(x)
 
        sh4 = x.shape
        x, ind4 = self.ma4(x)
        self.li4 = x = self.block4(x)
 
        sh5 = x.shape
        x, ind5 = self.ma5(x)
        self.li5 = x = self.block5(x)
 
        x = self.uma5(x, ind5, output_size = sh5)
        self.lo4 = x = self.blockUp4_a(x)
        self.lo4a = x = self.blockUp4_b(x + self.li4)

        x = self.uma(x, ind4, output_size = sh4)
        self.lo3 = x = self.blockUp3_a(x)
        self.lo3a = x = self.blockUp3_b(x + self.li3)


        x = self.uma(x, ind3, output_size = sh3)
        self.lo2 = x = self.blockUp2_a(x)
        self.lo2a = x = self.blockUp2_b(x + self.li2)

        x = self.uma(x, ind2, output_size = sh2)
        self.lo1 = x = self.blockUp1_a(x)
        self.lo1a = x = self.blockUp1_b(x + self.li1)

        #x = self.uma(x, ind1, output_size = sh1)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        self.lo0 =  x = self.blockUp0_a(x)
        self.lo0a = x = self.blockUp0_b1(x + self.li0)
        self.lo0b = x = self.blockUp0_b2(x)

        x = self.conv1x(x)
        return torch.sigmoid(x)

if 1:
    ac1 = Cap6Model()
    ac1.load_state_dict(torch.load(scriptpath + "/torchparams/paramsLRoiNo184_t1fs7_00105_train3800.pt", map_location=device), strict=False)
    ac1.to(device)
    ac1.eval()

OUTPUT_MORE = False

OUTPUT_RES64 = False
OUTPUT_NATIVE = True
OUTPUT_DEBUG = False

if "-v" in sys.argv:
    sys.argv.remove("-v")
    OUTPUT_MORE = True


allsubjects_accumulate_txt = []

mul_homo = lambda g, Mt : g @ Mt[:3,:3].astype(np.float32) + Mt[3,:3].astype(np.float32)

def indices_unitary(dimensions, dtype):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,)*N
    res = np.empty((N,)+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        res[i] = np.linspace(-1, 1, dim, dtype=dtype).reshape( shape[:i] + (dim,) + shape[i+1:]  )
    return res

for fname in sys.argv[1:]:
    Ti = time.time()
    try:
        print("Loading image " + fname)
        outfilename = fname.replace(".mnc", ".nii").replace(".mgz", ".nii").replace(".nii.gz", ".nii").replace(".nii", "_tiv.nii.gz")
        img = nibabel.load(fname)

        if type(img) is nibabel.nifti1.Nifti1Image:
            img._affine = img.get_qform() # for ANTs compatibility

        if type(img) is nibabel.Nifti1Image:
            if img.header["qform_code"] == 0:
                if img.header["sform_code"] == 0:
                    print(" *** Error: the header of this nifti file has no qform_code defined.")
                    print(" Fix the header manually or reconvert from the original DICOM.")
                    if not OUTPUT_DEBUG:
                        continue

            if not np.allclose(img.get_sform(), img.get_qform()):
                img._affine = img.get_qform() # simplify later ANTs compatibility
                print("This image has an sform defined, ignoring it - work in scanner space using the qform")

        d = img.get_fdata(caching="unchanged", dtype=np.float32)
    except:
        open(fname + ".warning.txt", "a").write("can't open the file\n")
        print(" *** Error: can't open file. Skip")
        continue

    intens_info = {"min":d.min(), "max":d.max(), "mean":d.mean(), "std":d.std()}
    while len(d.shape) > 3:
        print("Warning: this looks like a timeserie. Picking the first volume only")
        open(fname + ".warning.txt", "a").write("dim not 3. Picking first on last dimension\n")
        d = d[...,-1]

    d = (d - d.mean()) / d.std()


    ## For collecting Output user-statistics
    scalar_output = {}

    if 1:
        o1 = nibabel.orientations.io_orientation(img.affine)
        o2 = np.array([[ 0., -1.], [ 1.,  1.], [ 2.,  1.]]) # We work in LAS space (same as the mni_icbm152 template)
        trn = nibabel.orientations.ornt_transform(o1, o2) # o1 to o2 (apply to o2 to obtain o1)
        trn_back = nibabel.orientations.ornt_transform(o2, o1)    

        revaff1 = nibabel.orientations.inv_ornt_aff(trn, (1,1,1)) # mult on o1 to obtain o2
        revaff1i = nibabel.orientations.inv_ornt_aff(trn_back, (1,1,1)) # mult on o2 to obtain o1

        aff_orig64 = np.linalg.lstsq(bbox_world(np.identity(4), (64,64,64)), bbox_world(img.affine, img.shape[:3]), rcond=None)[0].T
        voxscale_native64 = np.abs(np.linalg.det(aff_orig64))
        revaff64i = nibabel.orientations.inv_ornt_aff(trn_back, (64,64,64))
        aff_reor64 = np.linalg.lstsq(bbox_world(revaff64i, (64,64,64)), bbox_world(img.affine, img.shape[:3]), rcond=None)[0].T

        wgridt = (netAff.grid @ torch.tensor(revaff1i, device=device, dtype=torch.float32))[None,...,[2,1,0]]
        d_orr = F.grid_sample(torch.as_tensor(d, dtype=torch.float32, device=device)[None,None], wgridt, align_corners=True)

        if OUTPUT_DEBUG:
            nibabel.Nifti1Image(np.asarray(d_orr[0,0].cpu()), aff_reor64).to_filename(outfilename.replace("_tiv", "_orig_b64"))

    ## Head priors
        T = time.time()
        with torch.no_grad():
            out1t = net(d_orr)
        out1 = np.asarray(out1t.cpu())
        #print("Head Inference in ", time.time() - T)

        # brain mask
        output = out1[0,0].astype("float32")

        out_cc, lab = scipy.ndimage.label(output > .01)
        #output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)
        brainmask_cc = torch.tensor(output)

        if 0:
            inner_02and98th = np.histogram(d_orr[0,0,output > .5], bins=50)[1][[1, -2]]
            intens_info["brain_min"] = intens_info["std"] * inner_02and98th[0] + intens_info["mean"]
            intens_info["brain_max"] = intens_info["std"] * inner_02and98th[1] + intens_info["mean"]
            rescale_u8 = lambda x : (x-intens_info["min"]) / (intens_info["max"]-intens_info["min"]) * 255
            intens_info["u8_brain_min"] = rescale_u8(intens_info["brain_min"])
            intens_info["u8_brain_max"] = rescale_u8(intens_info["brain_max"])

            txt = ",".join(intens_info) + "\n" + ",".join(["%f" % x for x in intens_info.values()]) + "\n"
            open(outfilename.replace("_tiv.nii.gz", "_intensity_info.csv"), "w").write(txt)

        vol = (output[output > .5]).sum() * voxscale_native64
        if OUTPUT_DEBUG:
            print(" Estimated intra-cranial volume (mm^3): %d" % vol)
        if 0:
            open(outfilename.replace("_tiv.nii.gz", "_eTIV.txt"), "w").write("%d\n" % vol)
        scalar_output["eTIV_mni"] = vol

        if OUTPUT_RES64:
            out = (output.clip(0, 1) * 255).astype("uint8")
            nibabel.Nifti1Image(out, aff_reor64, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d_b64" % 0))

        if OUTPUT_NATIVE:
            # wgridt for native space
            gsx, gsy, gsz = img.shape[:3]
            # this is a big array, so use float16
            sgrid = np.rollaxis(indices_unitary((gsx,gsy,gsz), dtype=np.float16),0,4)
            wgridt = torch.as_tensor(mul_homo(sgrid, inv(revaff1i))[None,...,[2,1,0]], device=device, dtype=torch.float32)
            del sgrid

            dnat = np.asarray(F.grid_sample(torch.as_tensor(output, dtype=torch.float32, device=device)[None,None], wgridt, align_corners=True).cpu())[0,0]
            #nibabel.Nifti1Image(dnat, img.affine).to_filename(outfilename.replace("_tiv", "_tissues%d" % 0))
            if OUTPUT_MORE:
                nibabel.Nifti1Image((dnat > .5).astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_brain_mask"))
            vol = (dnat > .5).sum() * np.abs(np.linalg.det(img.affine))
            print(" Estimated intra-cranial volume (mm^3) (native space): %d" % vol)
            scalar_output["eTIV"] = vol
            del dnat

        if 1:
            # cerebrum mask
            output = out1[0,2].astype("float32")
        
            out_cc, lab = scipy.ndimage.label(output > .01)
            output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)
        
            vol = (output[output > .5]).sum() * voxscale_native64
            if OUTPUT_DEBUG:
                print(" Estimated cerebrum volume (mm^3): %d" % vol)
            if 0:
                open(outfilename.replace("_tiv.nii.gz", "_eTIV_nocerebellum.txt"), "w").write("%d\n" % vol)
            scalar_output["cerebrum_vol_mni"] = vol
        
            if OUTPUT_RES64:
                out = (output.clip(0, 1) * 255).astype("uint8")
                nibabel.Nifti1Image(out, aff_reor64, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d_b64" % 2))
            if OUTPUT_NATIVE:
                dnat = np.asarray(F.grid_sample(torch.as_tensor(output, dtype=torch.float32, device=device)[None,None], wgridt, align_corners=True).cpu()[0,0])
                #nibabel.Nifti1Image(dnat, img.affine).to_filename(outfilename.replace("_tiv", "_tissues%d" % 2))
                if OUTPUT_MORE:
                    nibabel.Nifti1Image((dnat > .5).astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_cerebrum_mask"))
                vol = (dnat > .5).sum() * np.abs(np.linalg.det(img.affine))
                print(" Estimated cerebrum volume (mm^3) (native space): %d" % vol)
                scalar_output["cerebrum_vol"] = vol
                del dnat


        # cortex
        output = out1[0,1].astype("float32")
        output[output < .01] = 0
        if OUTPUT_RES64:
            out = (output.clip(0, 1) * 255).astype("uint8")
            nibabel.Nifti1Image(out, aff_reor64, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d_b64" % 1))
        if OUTPUT_NATIVE and OUTPUT_DEBUG:
            dnat = np.asarray(F.grid_sample(torch.as_tensor(output, dtype=torch.float32, device=device)[None,None], wgridt, align_corners=True).cpu()[0,0])
            nibabel.Nifti1Image(dnat, img.affine).to_filename(outfilename.replace("_tiv", "_tissues%d" % 1))
            del dnat


    ## MNI affine
        T = time.time()
        with torch.no_grad():
            wc1, tA = netAff(out1t[:,[1,3]] * brainmask_cc)

        wnat = np.linalg.lstsq(bbox_world(img.affine, img.shape[:3]), bbox_one @ revaff1, rcond=None)[0]
        wmni = np.linalg.lstsq(bbox_world(affine64_mni, (64,64,64)), bbox_one, rcond=None)[0]
        M = (wnat @ inv(np.asarray(tA[0].cpu())) @ inv(wmni)).T
        del wnat
        # [native world coord] @ M.T -> [mni world coord] , in LAS space

        if OUTPUT_DEBUG:
            # Output MNI, mostly for debug, save in box64, uint8
            out2 = np.asarray(wc1.to("cpu"))
            out2 = np.clip((out2 * 255), 0, 255).astype("uint8")
            nibabel.Nifti1Image(out2[0,0], affine64_mni).to_filename(outfilename.replace("_tiv", "_mniwrapc1"))
            del out2
        if 0:
            out2r = np.asarray(netAff.resample_other(d_orr).cpu())
            out2r = (out2r - out2r.min()) * 255 / out2r.ptp()
            nibabel.Nifti1Image(out2r[0,0].astype("uint8"), affine64_mni).to_filename(outfilename.replace("_tiv", "_mniwrap"))
            del out2r



        # output an ANTs-compatible matrix (AntsApplyTransforms -t)
        f3 = np.array([[1, 1, -1, -1],[1, 1, -1, -1], [-1, -1, 1, 1], [1, 1, 1, 1]]) # ANTs LPS
        MI = inv(M) * f3
        txt = """#Insight Transform File V1.0\nTransform: AffineTransform_float_3_3\nFixedParameters: 0 0 0\nParameters: """
        txt += " ".join(["%4.6f %4.6f %4.6f" % tuple(x) for x in MI[:3,:3].tolist()]) + " %4.6f %4.6f %4.6f\n" % (MI[0,3], MI[1,3], MI[2,3])
        if OUTPUT_MORE:
            open(outfilename.replace("_tiv.nii.gz", "_mni0Affine.txt"), "w").write(txt)

        u, s, vt = np.linalg.svd(MI[:3,:3])
        MI3rigid = u @ vt
        txt = """#Insight Transform File V1.0\nTransform: AffineTransform_float_3_3\nFixedParameters: 0 0 0\nParameters: """
        txt += " ".join(["%4.6f %4.6f %4.6f" % tuple(x) for x in MI3rigid.tolist()]) + " %4.6f %4.6f %4.6f\n" % (MI[0,3], MI[1,3], MI[2,3])
        if OUTPUT_MORE:
            open(outfilename.replace("_tiv.nii.gz", "_mni0Rigid.txt"), "w").write(txt)

    mniaffine_scaling = np.abs(np.linalg.det(M))
    #print(" mni affine scaling factor: ", mniaffine_scaling)

## Lesion
    T = time.time()
    imgcroproi_affine = np.array([[  -1. ,  -0. ,   0. ,  90.], [  -0. ,   1. ,  -0. ,-117.], [   0. ,   0.,   1.5,  -33.], [   0.,    0.,    0.,    1.]])
    imgcroproi_shape = (184, 202, 72)
    # coord in mm bbox
    sgrid = np.rollaxis(indices_unitary(imgcroproi_shape, dtype=np.float32),0,4)

    wnat = np.linalg.lstsq(bbox_world(img.affine, img.shape[:3]), bbox_one, rcond=None)[0]
    bboxnat = bbox_world(imgcroproi_affine, imgcroproi_shape) @ inv(M.T) @ wnat
    matzoom = np.linalg.lstsq(bbox_one, bboxnat, rcond=None)[0] # in -1..1 space
    wgridt = torch.tensor(mul_homo( sgrid, matzoom )[None,...,[2,1,0]], device=device, dtype=torch.float32)
    del sgrid, wnat
    dout = F.grid_sample(torch.as_tensor(d, dtype=torch.float32, device=device)[None,None], wgridt, align_corners=True)
    # note: d was normalized from full-image

    if OUTPUT_DEBUG:
        d_in = np.asarray(dout[0,0].cpu()) # back to numpy since torch does not support negative step/strides
        d_in_u8 = (((d_in - d_in.min()) / d_in.ptp()) * 255).astype("uint8")
        nibabel.Nifti1Image(d_in_u8, imgcroproi_affine).to_filename(outfilename.replace("_tiv", "_afcrop"))

    d_in = dout
    d_in -= d_in.mean()
    d_in /= d_in.std()
    
    with torch.no_grad():
        outseg = ac(torch.as_tensor(d_in)) #[:,:1] # second feature map is redundant
        outcortex = outseg[:,1:2]
        outseg = outseg[:,0:1]

    #print("Inferrence in " + str(time.time() - T))

    if OUTPUT_DEBUG:
        output = np.asarray(outseg[0, 0].cpu())
        output[output < .1] = 0 # remove noise, mostly for better compression
        outputfn = outfilename.replace("_tiv", "_afcrop_outseg_wmh")
        nibabel.Nifti1Image(np.clip(output * 255, 0, 255).astype(np.uint8), imgcroproi_affine).to_filename(outputfn)

    # smoothly rescale (.5 ~ .75) to (.5 ~ 1.) # This impact strongly the absolute value of the result    
    outseg = torch.clamp((outseg - .5) * 2 + .5, 0, 1) * (outseg > .5)
    
    if 1:

        def bbox_xyz(shape, affine):
            " returns the worldspace of the edge of the image "
            s = shape[0]-1, shape[1]-1, shape[2]-1
            bbox = [[0,0,0], [s[0],0,0], [0,s[1],0], [0,0,s[2]], [s[0],s[1],0], [s[0],0,s[2]], [0,s[1],s[2]], [s[0],s[1],s[2]]]
            return mul_homo(bbox, affine.T)

        def indices_xyz(shape, affine, offset_vox= np.array([0,0,0])):
            assert (len(shape) == 3)
            ind = np.indices(shape).astype(np.float32) + offset_vox.reshape(3, 1,1,1).astype(np.float32)
            return mul_homo(np.rollaxis(ind, 0, 4), affine.T)

        def xyz_to_DHW3(xyz, iaffine, srcshape):
            affine = np.linalg.inv(iaffine)
            ijk3 = mul_homo(xyz, affine.T)
            ijk3[...,0] /= srcshape[0] -1
            ijk3[...,1] /= srcshape[1] -1
            ijk3[...,2] /= srcshape[2] -1
            ijk3 = ijk3 * 2 - 1
            DHW3 = np.swapaxes(ijk3, 0, 2)
            return DHW3

        pts = bbox_xyz(imgcroproi_shape, imgcroproi_affine)
        pts = mul_homo(pts, np.linalg.inv(M).T)
        pts_ijk = mul_homo(pts, np.linalg.inv(img.affine).T)
        for i in range(3):
            np.clip(pts_ijk[:,i], 0, img.shape[i], out = pts_ijk[:,i])
        pmin = np.floor(np.min(pts_ijk, 0)).astype(int)
        pwidth = np.ceil(np.max(pts_ijk, 0)).astype(int) - pmin

        widx = indices_xyz(pwidth, img.affine, offset_vox=pmin)

        widx = mul_homo(widx, M.T)

        DHW3 = xyz_to_DHW3(widx, imgcroproi_affine, imgcroproi_shape)

        wdata = np.zeros(img.shape[:3], np.uint8)

        d = outseg[0, 0].permute(2, 1, 0)
        outDHW = F.grid_sample(d[None,None], torch.tensor(DHW3[None]), align_corners=True)
        dnat = np.asarray(outDHW[0,0].permute(2,1,0))
        dnat[dnat < .1] = 0 # remove noise
        wdata[pmin[0]:pmin[0]+pwidth[0], pmin[1]:pmin[1]+pwidth[1], pmin[2]:pmin[2]+pwidth[2]] = (dnat * 255).clip(0, 255).astype(np.uint8)
        if OUTPUT_MORE:
            nibabel.Nifti1Image(wdata.astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_prob_wmh"))
        nibabel.Nifti1Image((wdata >= 128).astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_mask_wmh"))
        
        # use a PVE interpratation of probabilities at region borders to compute the full volume
        # remap the ConvNet values (0~.75 to 0~1) to sharpen the histogram into more voxels at exact 1.
        wmh_vol_native = (wdata.clip(0, 192) / 192. ).sum() * np.abs(np.linalg.det(img.affine))
        print(" WMH vol (mm^3) (native space): ", wmh_vol_native)
        #dnatL = dnat.astype(np.uint8).copy() # keep for screenshot

        scalar_output["wmh_vol"] = wmh_vol_native

    if OUTPUT_DEBUG:
        d = outcortex[0, 0].T
        outDHW = F.grid_sample(d[None,None], torch.tensor(DHW3[None]), align_corners=True)
        dnat = np.asarray(outDHW[0,0].T)
        dnat[dnat < .5] = 0 # remove noise
        wdata[pmin[0]:pmin[0]+pwidth[0], pmin[1]:pmin[1]+pwidth[1], pmin[2]:pmin[2]+pwidth[2]] = (dnat * 255).clip(0, 255).astype(np.uint8)
        nibabel.Nifti1Image(wdata.astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_corticalribbon"))
        del dnat

    wmh_map = np.clip(np.asarray(outseg.cpu()) * 255, 0, 255).astype(np.uint8)
    del outseg

# Lesions statistics - generate mask and use it for summing
    if 1:
        Tiroi = time.time()
        with torch.no_grad():
            outrois = ac1(torch.as_tensor(d_in))
            r, t = outrois.max(axis=1)
            labelmap = (t + 1).to(torch.float32)
            labelmap[r < .25] = 0
            labelmap = np.asarray(labelmap[0])
            del r, t

        if OUTPUT_DEBUG:
            outputfn = outfilename.replace("_tiv", "_afcrop_labelmap_%d" % i)
            nibabel.Nifti1Image(labelmap.astype(np.uint8), imgcroproi_affine).to_filename(outputfn)

        roisize = scipy.ndimage.sum(1, labels=labelmap, index=[1,2,3,4])
        voxvol = np.abs(np.linalg.det(imgcroproi_affine))
        scalar_output["roi_vol"] = roisize * voxvol / mniaffine_scaling
                
        wmh_lesions_vox_per_roi = scipy.ndimage.sum((wmh_map > 128) * wmh_map, labels=labelmap, index=[1,2,3,4]) / 255.
        wmh_lesions_vox_per_roi_std = np.sqrt(scipy.ndimage.variance((wmh_map > 128) * wmh_map, labels=labelmap, index=[1,2,3,4])) / 255.
        voxvol = np.abs(np.linalg.det(imgcroproi_affine))
        scalar_output["wmh_vol_per_roi"] = wmh_lesions_vox_per_roi * voxvol / mniaffine_scaling
        #scalar_output["wmh_vol_per_roi_std"] = wmh_lesions_vox_per_roi_std * voxvol / mniaffine_scaling

        d = torch.from_numpy(labelmap).to(torch.float32).permute(2,1,0)
        outDHW = F.grid_sample(d[None,None], torch.tensor(DHW3[None]), align_corners=True, mode="nearest")
        dnat = np.asarray(outDHW[0,0].permute(2,1,0))
        wdata[pmin[0]:pmin[0]+pwidth[0], pmin[1]:pmin[1]+pwidth[1], pmin[2]:pmin[2]+pwidth[2]] = dnat
        nibabel.Nifti1Image(wdata, img.affine).to_filename(outfilename.replace("_tiv", "_mask_ROIs"))
        #print("ROI time %4.2fs" % (time.time() - Tiroi))

    if 1:
        #txt = "eTIV,wmh_vol,deep,white,peri,deep_sd,white_sd,peri_sd,num_infarct\n"
        txtH = "eTIV,wmh_vol_total,wmh_vol_periventricular,wmh_vol_deepwhite,wmh_vol_infracortical,wmh_vol_innergray,roisize_periventricular,roisize_deepwhite,roisize_infracortical,roisize_innergray\n"
        txt = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % ((
            scalar_output.get("eTIV_mni", 0),
            scalar_output.get("wmh_vol")) +
            tuple(scalar_output.get("wmh_vol_per_roi")) +
            tuple(scalar_output.get("roi_vol"))
            )

        open(outfilename.replace("_tiv.nii.gz", "_wmh_in_lrois.csv"), "w").write(txtH + txt)
        print(" Saving scalars stats as", outfilename.replace("_tiv.nii.gz", "_wmh_in_lrois.csv"))
        
    print(" Elapsed time for subject %4.2fs " % (time.time() - Ti))
    #print(" To display using fslview, try:")
    #print("  fslview %s %s -t .5 %s -t .5 &" % (fname, outfilename.replace("_tiv", "_mask_wmh"), outfilename.replace("_tiv", "_mask_ROIs")))


    allsubjects_accumulate_txt.append(fname + "," + txt)


if 1: #OUTPUT_DEBUG:
    print("Peak memory used (Gb) " + str(resource.getrusage(resource.RUSAGE_SELF)[2] / (1024.*1024)))

#print("Done")

if len(sys.argv[1:]) > 1:
    outfilename = (os.path.dirname(fname) or ".") + "/all_subjects_wmh_report.csv"
    open(outfilename, "w").writelines( ["filename," + txtH] + allsubjects_accumulate_txt )
    print("Done\nVolumes of all subjects summarized as " + outfilename)
