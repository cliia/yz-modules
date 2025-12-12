import numpy as np
import cupy as cp
import time
import scipy.io as sio

# ====== 定义CUDA内核函数 ====== #
# 更新重建体积，应用非负约束
update_rec_kernel = cp.RawKernel(r'''
extern "C" __global__ 
void update_rec(float *rec_new, int dimx, int dimy) {
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    if (x < dimx) {
        long long i = (long long)z*dimx*dimy + y*dimx + x;
        rec_new[i] = max(0.0f, rec_new[i]);
    }
}
''', 'update_rec')

# 计算旋转偏移
compute_xy_shift_kernel = cp.RawKernel(r'''
extern "C" __global__ 
void compute_xy_shift(const float *Matrix, const float *shift, float *x_shift, float *y_shift, int Num_pjs) {
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < Num_pjs) {
        int index = 9 * i;
        for (int j = 0; j < 4; j++) {
            x_shift[4*i+j] = Matrix[index+0]*shift[2*j] + Matrix[index+3]*0.0f + Matrix[index+6]*shift[2*j+1];
            y_shift[4*i+j] = Matrix[index+1]*shift[2*j] + Matrix[index+4]*0.0f + Matrix[index+7]*shift[2*j+1];
        }
    }
}
''', 'compute_xy_shift')

# 计算残差
compute_residual_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_residual(float *residual, const float *projections, const float scale, long long N) {
    long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        residual[i] = residual[i] * scale - projections[i];
    }
}
''', 'compute_residual')

# 前向投影
radon_tf_kernel = cp.RawKernel(r'''
extern "C" __global__
void radon_tf(const float *data, const float *Matrix, const int nrows, const int ncols, const float *nc, 
               const int o_ratio, const float *x_s, const float *y_s, float *result, int dimz) {
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    
    int origin_offset = 1;
    long s;
    
    if (x < nrows) {
        const float data_i = data[z*ncols*nrows + y*nrows + x];
        const float coord_x = x - nc[0] + 1;
        const float coord_y = y - nc[1] + 1;
        const float coord_z = z - nc[2] + 1;

        const float x_i = Matrix[0]*coord_x + Matrix[3]*coord_y + Matrix[6]*coord_z + nc[0];
        const float y_i = Matrix[1]*coord_x + Matrix[4]*coord_y + Matrix[7]*coord_z + nc[1];

        for (s=0; s<o_ratio; s++) {
            float x_is = x_i + x_s[s] - origin_offset;
            float y_is = y_i + y_s[s] - origin_offset;

            // 获取边界网格位置
            long long x_1 = (long long)floor(x_is);
            long long x_2 = x_1 + 1;
            long long y_1 = (long long)floor(y_is);
            long long y_2 = y_1 + 1;
            
            if (x_1 >= -1 && x_2 <= nrows && y_1 >= -1 && y_2 <= ncols) { 
                float w_x1 = x_2 - x_is;
                float w_x2 = 1 - w_x1;
                float w_y1 = y_2 - y_is;
                float w_y2 = 1 - w_y1;
                
                if (x_1 == -1) {
                    if (y_1 == -1) {
                        atomicAdd(&result[x_2 + y_2*nrows], w_x2*w_y2 * data_i);
                    }
                    else if (y_2 == ncols) {
                        atomicAdd(&result[x_2 + y_1*nrows], w_x2*w_y1 * data_i);
                    }
                    else {
                        atomicAdd(&result[x_2 + y_1*nrows], w_x2*w_y1 * data_i);
                        atomicAdd(&result[x_2 + y_2*nrows], w_x2*w_y2 * data_i);                    
                    }
                }
                else if (x_2 == nrows) {
                    if (y_1 == -1) {
                        atomicAdd(&result[x_1 + y_2*nrows], w_x1*w_y2 * data_i);
                    }
                    else if (y_2 == ncols) {
                        atomicAdd(&result[x_1 + y_1*nrows], w_x1*w_y1 * data_i);
                    }
                    else {
                        atomicAdd(&result[x_1 + y_1*nrows], w_x1*w_y1 * data_i);
                        atomicAdd(&result[x_1 + y_2*nrows], w_x1*w_y2 * data_i);                  
                    } 
                }
                else {
                    if (y_1 == -1) {
                        atomicAdd(&result[x_1 + y_2*nrows], w_x1*w_y2 * data_i);
                        atomicAdd(&result[x_2 + y_2*nrows], w_x2*w_y2 * data_i);
                    }
                    else if (y_2 == ncols) {
                        atomicAdd(&result[x_1 + y_1*nrows], w_x1*w_y1 * data_i);
                        atomicAdd(&result[x_2 + y_1*nrows], w_x2*w_y1 * data_i);
                    }
                    else {
                        atomicAdd(&result[x_1 + y_1*nrows], w_x1*w_y1 * data_i);
                        atomicAdd(&result[x_1 + y_2*nrows], w_x1*w_y2 * data_i);
                        atomicAdd(&result[x_2 + y_1*nrows], w_x2*w_y1 * data_i);
                        atomicAdd(&result[x_2 + y_2*nrows], w_x2*w_y2 * data_i);                  
                    }                               
                }
            }
        }
    }
}
''', 'radon_tf')

# 反投影
radon_tpose_kernel = cp.RawKernel(r'''
extern "C" __global__
void radon_tpose(const float *Matrix, const int nrows, const int ncols, const float *nc, const float *data,
                 const int o_ratio, const float *x_s, const float *y_s, float *Rec, float dt, long long N) {
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    int origin_offset = 1;
    long s;
      
    if (x < nrows) {
        const float coord_x = x - nc[0] + 1;
        const float coord_y = y - nc[1] + 1;
        const float coord_z = z - nc[2] + 1;
        long long i = (long long)z*ncols*nrows + y*nrows + x;

        const float x0 = Matrix[0]*coord_x + Matrix[3]*coord_y + Matrix[6]*coord_z + nc[0];
        const float y0 = Matrix[1]*coord_x + Matrix[4]*coord_y + Matrix[7]*coord_z + nc[1];
        
        for (s=0; s<o_ratio; s++) {
            float x_i = x0 + x_s[s];
            float y_i = y0 + y_s[s];
            
            // 获取边界网格位置
            long long x_1 = (long long)floor(x_i) - origin_offset;
            long long x_2 = x_1 + 1;
            long long y_1 = (long long)floor(y_i) - origin_offset;
            long long y_2 = y_1 + 1;
            
            // 处理x/y是最后一个元素的特殊情况
            if ((x_i - origin_offset) == (nrows-1)) { x_2 -= 1; x_1 -= 1; }
            if ((y_i - origin_offset) == (ncols-1)) { y_2 -= 1; y_1 -= 1; }
            
            // 对于超出边界的目标值返回0
            if (x_1 < 0 || x_2 > (nrows - 1) || y_1 < 0 || y_2 > (ncols - 1)) {
                // 不做任何操作
            }
            else {
                // 获取数组值
                const float f_11 = data[x_1 + y_1*nrows];
                const float f_12 = data[x_1 + y_2*nrows];
                const float f_21 = data[x_2 + y_1*nrows];
                const float f_22 = data[x_2 + y_2*nrows];
                
                // 计算权重
                float w_x1 = x_2 - (x_i - origin_offset);
                float w_x2 = (x_i - origin_offset) - x_1;
                float w_y1 = y_2 - (y_i - origin_offset);
                float w_y2 = (y_i - origin_offset) - y_1;
                
                float a, b;
                a = f_11 * w_x1 + f_21 * w_x2;
                b = f_12 * w_x1 + f_22 * w_x2;
                Rec[i] -= dt * (a * w_y1 + b * w_y2);
            }
        }
    }
}
''', 'radon_tpose')

# 计算L1范数的核函数
sum_abs_kernel = cp.RawKernel(r'''
extern "C" __global__
void sum_abs(const float *gArr, long long arraySize, float *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * 1024;
    const int gridSize = 1024 * gridDim.x;
    float sum = 0;
    
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += fabsf(gArr[i]);
        
    __shared__ float shArr[1024];
    shArr[thIdx] = sum;
    __syncthreads();
    
    for (int size = 512; size > 0; size /= 2) {
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}
''', 'sum_abs')

# ====== 三维重建函数 ====== #
def resire_reconstruct(projections, matrix, dimz, iterations=100, step_size=1.0, positivity=True, initial_rec=None):
    """
    使用CuPy实现的三维重建函数，模仿RT3_1GPU_modified.cu功能
    
    参数:
    projections: 投影数据，形状为(dimx, dimy, num_pjs)的numpy数组
    matrix: 旋转矩阵，形状为(3, 3, num_pjs)的numpy数组
    dimz: 重建体积的z维度大小
    iterations: 重建迭代次数
    step_size: 重建步长
    positivity: 是否强制非负约束
    initial_rec: 初始重建，如果为None则从零开始
    
    返回:
    rec: 重建的3D体积，形状为(dimx, dimy, dimz)的numpy数组
    r_value: 最终R1范数值
    """
    # 确保输入数据是单精度浮点数
    if projections.dtype != np.float32:
        projections = projections.astype(np.float32)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    
    # 获取维度信息
    dimx, dimy, Num_pjs = projections.shape
    
    # 计算各种大小
    o_ratio = 4  # 与原CUDA代码相同的子像素采样数
    nrow_cols = dimx * dimy
    nPjsPoints = dimx * dimy * Num_pjs
    recPoints = dimx * dimy * dimz
    
    # 计算中心点 - 与原始代码保持一致
    ncx = (dimx + 1.0) / 2.0
    ncy = (dimy + 1.0) / 2.0
    ncz = (dimz + 1.0) / 2.0
    d_nc = cp.array([float(ncx), float(ncy), float(ncz)], dtype=cp.float32)
    
    # 修改: 注意原始CUDA代码中，投影数据的内存布局
    # 在CUDA中，数据排列是行优先的，所以需要确保与之一致
    # projections形状为(dimx, dimy, Num_pjs)
    # 在CUDA中索引方式为: d_projections[i*nrow_cols + y*dimx + x]
    d_projections = cp.asarray(projections.transpose(2, 1, 0).reshape(-1))
    
    # 确保矩阵内存布局与原始CUDA代码一致
    # 在原始CUDA代码中，矩阵以行为主序，每个角度9个元素
    d_matrix_flat = cp.asarray(matrix.transpose(2, 1, 0).reshape(-1))
    
    # 创建重建体积 - 使用与CUDA代码一致的内存布局
    # 在CUDA中，体积索引方式为: d_rec[z*dimy*dimx + y*dimx + x]
    if initial_rec is not None:
        if initial_rec.shape == (dimx, dimy, dimz):
            # 转置为与CUDA索引一致的顺序(z,y,x)
            d_rec = cp.asarray(initial_rec.transpose(2, 1, 0).reshape(-1))
        else:
            raise ValueError("初始重建形状必须为(dimx, dimy, dimz)")
    else:
        d_rec = cp.zeros(recPoints, dtype=cp.float32)
    
    # 创建残差 - 与CUDA代码一致的内存布局
    d_residual = cp.zeros(nPjsPoints, dtype=cp.float32)
    
    # 确保步长计算正确
    dt = np.float32(step_size / Num_pjs / dimz / o_ratio)
    
    # 确保偏移计算正确
    shift = cp.array([0.25, 0.25, 0.25, -0.25, -0.25, 0.25, -0.25, -0.25], dtype=cp.float32)
    x_shift = cp.zeros(4 * Num_pjs, dtype=cp.float32)
    y_shift = cp.zeros(4 * Num_pjs, dtype=cp.float32)
    
    # 调用计算旋转偏移的核函数
    threadsPerBlock = 256
    blocksPerGrid = (Num_pjs + threadsPerBlock - 1) // threadsPerBlock
    compute_xy_shift_kernel((blocksPerGrid,), (threadsPerBlock,), 
                           (d_matrix_flat, shift, x_shift, y_shift, np.int32(Num_pjs)))
    
    # # 调试输出检查偏移计算
    # first_shift_x = cp.asnumpy(x_shift[0:4])
    # first_shift_y = cp.asnumpy(y_shift[0:4])
    
    # 设置CUDA核函数调用的块和线程配置
    blockSize = 1024
    gridSize = 24
    blocksPerGridPrj = (nPjsPoints + threadsPerBlock - 1) // threadsPerBlock
    blocksPerGridRec = (recPoints + threadsPerBlock - 1) // threadsPerBlock
    
    # 使用3D网格进行前向和反向投影，确保维度顺序正确
    # 在CUDA中，块索引是: (x, y, z) 对应 (dimx, dimy, dimz)
    blocksPerGridRec2 = ((dimx + threadsPerBlock - 1) // threadsPerBlock, dimy, dimz)
    
    # 计算投影的L1范数
    sum_buffer = cp.zeros(gridSize, dtype=cp.float32)
    sum_abs_kernel((gridSize,), (blockSize,), (d_projections, nPjsPoints, sum_buffer))
    sum_abs_kernel((1,), (blockSize,), (sum_buffer, gridSize, sum_buffer))
    pj_norm = float(sum_buffer[0])
    start_time = time.time()
    
    # 迭代重建
    for iter in range(iterations):
        # 清零残差
        d_residual.fill(0)
        
        # 前向投影 - 确保索引与CUDA代码一致
        for i in range(Num_pjs):
            radon_tf_kernel(
                blocksPerGridRec2, 
                (threadsPerBlock, 1, 1), 
                (d_rec, d_matrix_flat[9*i:9*(i+1)], np.int32(dimx), np.int32(dimy), 
                 d_nc, np.int32(o_ratio), x_shift[i*o_ratio:(i+1)*o_ratio], 
                 y_shift[i*o_ratio:(i+1)*o_ratio], d_residual[i*nrow_cols:(i+1)*nrow_cols], np.int32(dimz))
            )
        
        # 计算残差: 前向投影 - 测量投影
        compute_residual_kernel(
            (blocksPerGridPrj,), 
            (threadsPerBlock,), 
            (d_residual, d_projections, np.float32(1.0/o_ratio), np.int64(nPjsPoints))
        )
        
        # 每10次迭代计算R1因子
        if iter % 10 == 0 or iter == iterations - 1:
            sum_buffer.fill(0)
            sum_abs_kernel((gridSize,), (blockSize,), (d_residual, nPjsPoints, sum_buffer))
            sum_abs_kernel((1,), (blockSize,), (sum_buffer, gridSize, sum_buffer))
            res_norm = float(sum_buffer[0])
            print(f"{iter+1}. R1 = {res_norm/pj_norm}")
        
        # 反投影 - 确保索引与CUDA代码一致
        for i in range(Num_pjs):
            radon_tpose_kernel(
                blocksPerGridRec2, 
                (threadsPerBlock, 1, 1), 
                (d_matrix_flat[9*i:9*(i+1)], np.int32(dimx), np.int32(dimy), d_nc, d_residual[i*nrow_cols:(i+1)*nrow_cols],
                 np.int32(o_ratio), x_shift[i*o_ratio:(i+1)*o_ratio], y_shift[i*o_ratio:(i+1)*o_ratio], 
                 d_rec, dt, np.int64(recPoints))
            )
        
        # 如果需要应用非负约束
        if positivity:
            update_rec_kernel(
                blocksPerGridRec2, 
                (threadsPerBlock, 1, 1), 
                (d_rec, np.int32(dimx), np.int32(dimy))
            )
    
    # 计算最终R1因子
    sum_buffer.fill(0)
    sum_abs_kernel((gridSize,), (blockSize,), (d_residual, nPjsPoints, sum_buffer))
    sum_abs_kernel((1,), (blockSize,), (sum_buffer, gridSize, sum_buffer))
    res_norm = float(sum_buffer[0])
    r_value = res_norm / pj_norm
    
    end_time = time.time()
    print(f"重构用时: {end_time - start_time:.2f}秒")
    print(f"最终 R1 = {r_value}")
    
    # 将结果从GPU转回CPU并重新整形为3D体积
    # 首先reshape为(z,y,x)顺序，然后转置为(x,y,z)顺序
    # 这确保返回的体积与输入的投影维度一致
    rec_3d = cp.asnumpy(d_rec).reshape(dimz, dimy, dimx)
    rec = rec_3d.transpose(2, 1, 0)  # 从(z,y,x)转为(x,y,z)
    
    return rec, r_value


if __name__ == "__main__":
    """示例用法"""
    from yzaet import eul2aetrotm

    # 创建测试数据
    projections = sio.loadmat('../data/GENFIREprojections.mat')['FinalGEproj1'].astype(np.float32)
    angles = sio.loadmat('../data/GENFIREAngles.mat')['anglesforREconstruct']
    num_pjs = angles.shape[0]

    # 创建随机旋转矩阵
    matrix = eul2aetrotm(angles).astype(np.float32)
    print(matrix.shape)
    
    # 运行重建
    rec, r_value = resire_reconstruct(
        projections=projections,
        matrix=matrix,
        dimz=200,
        iterations=201,  # 少量迭代用于演示
        step_size=2.0,
        positivity=True
    )

    sio.savemat('../data/GENFIREreconstruction.mat', {'reconstruction': rec}) 

