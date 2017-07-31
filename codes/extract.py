# -*- coding: utf-8 -*-


def calculate(mm, *args, **kwargs):

    from numpy import allclose, zeros, empty, array, ones, stack
    from numpy import zeros_like, transpose, triu_indices, ones_like
    from numpy import argmax, indices, unique, vstack, save, hstack
    from numpy import newaxis, pi, inf, arccos,isfinite,arange,cross,tril_indices,triu_indices
    from numpy.linalg import inv, norm
    from scipy import ndimage

    searchsize = kwargs.get('searchsize', 2)

    sigmas = arange(0, 3.1, 0.1)

    # Build empty matrixs
    sigmal=len(sigmas)
    shape=list(mm.shape)
    m=zeros([sigmal]+list(shape))
    newshape=list(m.shape)
    infos=empty(newshape+[8],dtype=float)
    lamda=empty(newshape+[2],dtype=float)
    eigv=empty(newshape+[2,2],dtype=float)
    topoint=empty(newshape+[2,2],dtype=float)

    # Build Convolve matrixs
    index=array(ones((searchsize*2+1,searchsize*2+1)).nonzero())
    tempinfo=empty((index.shape[1],6),dtype=float)
    tempinfo[:,0]=1.
    tempinfo[:,1]=index[0]-(searchsize)
    tempinfo[:,2]=index[1]-(searchsize)
    tempinfo[:,3]=0.5*tempinfo[:,1]*tempinfo[:,1]
    tempinfo[:,4]=0.5*tempinfo[:,2]*tempinfo[:,2]
    tempinfo[:,5]=tempinfo[:,1]*tempinfo[:,2]
    tempinfo=((inv(tempinfo.T.dot(tempinfo)).dot(tempinfo.T)))
    tempinfo.resize(6,searchsize*2+1,searchsize*2+1)
    # Convolve need the filter to be inversed
    tempinfo=tempinfo[:,::-1,::-1]

    # Calculate Convolved matrixs
    for i,s in enumerate(sigmas):
        m[i]=ndimage.gaussian_filter(mm,s,mode='mirror')
        for j in range(1,6):
            infos[i,:,:,j-1]=ndimage.convolve(m[i],tempinfo[j],mode='nearest')+1e-100
        infos[i,:,:,5]=infos[i,:,:,2]+infos[i,:,:,3] # xx+yy
        infos[i,:,:,6]=infos[i,:,:,2]-infos[i,:,:,3] # xx-yy
        infos[i,:,:,7]=2*infos[i,:,:,4]              # 2xy
        lamda[i,:,:,0]=(infos[i,:,:,5]-(infos[i,:,:,6]**2+infos[i,:,:,7]**2)**.5)/2
        lamda[i,:,:,1]=(infos[i,:,:,5]+(infos[i,:,:,6]**2+infos[i,:,:,7]**2)**.5)/2
        temp=((lamda[i,:,:]-infos[i,:,:,3:4])/infos[i,:,:,4:5])
        eigv[i,:,:,0,:]=temp/(temp**2+1)**.5
        eigv[i,:,:,1,:]=1./(temp**2+1)**.5
        A=eigv[i,:,:,:,:1]*lamda[i,:,:,0][:,:,newaxis,newaxis]
        A.resize(shape+[2])
        b=-(eigv[i,:,:,:,0]*infos[i,:,:,:2]).sum(2)
        topoint[i,:,:,1,0]=A[:,:,1]
        topoint[i,:,:,1,1]=-A[:,:,0]
        tonorm=norm(topoint[i,:,:,1],axis=2,keepdims=True)
        tonorm[tonorm==0]=1
        topoint[i,:,:,1]/=tonorm
        temp=stack((A,topoint[i,:,:,1]),2)
        temp[norm(topoint[i,:,:,1],axis=2)!=0]=inv(temp[norm(topoint[i,:,:,1],axis=2)!=0])
        temp1=stack((b,zeros_like(b)),2)
        temp1.resize(list(temp1.shape)+[1])
        topoint[i,:,:,0]=(temp*transpose(temp1,(0,1,3,2))).sum(3)
        topoint[i,norm(topoint[i,:,:,1],axis=2)==0,0]=inf

    topoint[topoint[:,:,:,1,1]<0,1]*=-1

    # Basic filter: close to ridge and is ridge
    closeok=((topoint[:,:,:,0]**2).sum(3)<2)*(lamda[:,:,:,0]<0)

    mask=ones_like(mm)

    errors=infos[[slice(None,None)]+list(mask.nonzero())+[slice(5,8)]].std(axis=1,ddof=1)**2
    strength=(infos[:,:,:,5:8]**2-errors[:,newaxis,newaxis,:]).clip(min=0)
    strength[closeok==False]=0
    strength=strength[:,:,:,0]*(strength[:,:,:,1]+strength[:,:,:,2])*(sigmas[:,newaxis,newaxis]**6.)

    argm=argmax(strength,axis=0)
    xind,yind=indices(argm.shape)
    dewidth=sigmas[argm]
    dedeg=topoint[argm,xind,yind,1].copy()
    dedeg[dedeg[:,:,1]<0]*=-1
    dedeg=arccos(dedeg[:,:,0].clip(min=-1,max=1))/pi*180
    dedist=cross(topoint[argm,xind,yind,0],topoint[argm,xind,yind,1])
    deheight=(strength[argm,xind,yind]*64*dewidth**2)**.25

    temp=stack((deheight,dewidth,dedeg),axis=-1)
    temp[temp[:,:,0]==0,1]=0
    temp[temp[:,:,0]==0,2]=0

    return temp


def extract_ridge_feature(matrix, helixs):
    from numpy import arange, ones, zeros, isfinite, array, hstack, vstack
    deletecolumn = arange(5)

    allmatrix = zeros([len(deletecolumn)] + list(matrix.shape) + [3])

    for delete in deletecolumn:
        for i in range(delete + 1):
            for j in range(delete + 1):
                allmatrix[delete, i::delete + 1, j::delete + 1] = calculate(
                    matrix[i::delete + 1, j::delete + 1].copy())

    allfeature=[]
    for i in range(len(helixs)):
        for j in range(i+1,len(helixs)):
            tempmatrix=allmatrix[:,helixs[i][0]:helixs[i][1]+1,helixs[j][0]:helixs[j][1]+1]
            denoisepairfeature=[]
            for delete in deletecolumn:
                deletefeature=[]
                for ii in range(delete + 1):
                    deletefeature.append([])
                    for jj in range(delete + 1):
                        here=tempmatrix[delete,ii::delete+1,jj::delete+1].copy()
                        tempmask=isfinite(here[:,:,0])
                        if tempmask.sum()==0:
                            deletefeature[-1].append(zeros(3))
                        else:
                            deletefeature[-1].append(here[tempmask,:3][here[tempmask,0].argmax()])
                deletefeature=array(deletefeature)
                if deletefeature.shape!=(delete+1,delete+1,3):
                    raise ValueError('Wrong size')
                denoisepairfeature.append(deletefeature[:,:,0].flatten())
                denoisepairfeature.append(deletefeature[:,:,1].flatten())
                denoisepairfeature.append(deletefeature[:,:,2].flatten())
            denoisepairfeature=hstack(denoisepairfeature)
            allfeature.append(denoisepairfeature)

    allfeature=vstack(allfeature)

    return allfeature
