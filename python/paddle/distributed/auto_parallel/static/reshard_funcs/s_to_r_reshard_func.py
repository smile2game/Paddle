# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle

from ..process_group import new_process_group
from .base_reshard_func import (
    ReshardFunction,
    copy_op_attr_with_new_member,
    is_replicated,
    is_shard,
)
from .same_status_reshard_func import SameStatusReshardFunction


class SToRReshardFunction(ReshardFunction): #shard->replicated
    def is_suitable(self, src_dist_attr, dst_dist_attr): 
        if not is_shard(src_dist_attr): #起始是 shard
            return False

        if not is_replicated(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh.ndim != 1:
            return False
        if out_mesh.ndim != 1:
            return False
        if in_mesh != out_mesh:
            return False
        return True

    def infer_allgather_dist_type(self, in_value, split_axis): #推断all_gather之后的数据类型
        """
        infer dist type after allgather

        Args:
            in_value(tensor) : input tensor
            split_axis(int) : input tensor's split axis

        Returns:
            out_type(dist_type) : 输出的张量

        Examples: 
            假设输入为：
            in_value = [[1,2,3,4,5],
                        [6,7,8,9,10]]
            split_axis = 1

            (mesh = [0,1,2])
            (placements1 = [dist.Shard(1)])

            Steps:
            1. 获取输入张量的 维度数量/分布式属性(dims_mapping[split_axis]/mesh)    Note: 分布式属性 <> placements
                tensor_dim = #2
                in_dist_attr = #(dims_mapping,process_mesh)
                split_mesh_dim = #in_dist_attr.dims_mapping[1] = 0 表示张量的内层,在被mesh的0轴切分 (这里和placements的[dist.Shard(0)]不一样,placements代表去切分张量的第0轴)
                mesh = #[0,1,2]

            2. 推测out的 local_shape/global_shape/type     Note:一个Process只能看到自身的local shape,能够看到global tensor吗？
                out_local_shape = #[2,5]
                out_local_shape = #[2,5+3-1/3] = [2,2] #切分后的
                out_global_shape = #[2,2]
                out_global_shape = #[2,2*3] ?为什么是0维度相乘,难道不是[split_axis]吗？

                #Note:静态图需要显式定义 张量形状和数据类型，分布式还需要tensor_dist_attr
                out_type = paddle.pir.create_shaped_type(in_value.type,out_global_shape) #创建一个新的张量类型,数据结构和in_value相同,形状和全局输出相同,这是静态图必须的

            3. 推测out的 dims_mapping/dist_attr/type
                out_dims_mapping = #[-1,0]
                out_dims_mapping = #[-1,-1] #现在搞成 replicated了,不再是 shard了
                
                #Note:tensor_dist_attr包含 mesh,dims_mapping,partial_status,
                out_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute( mesh, out_dims_mapping, in_dist_attr.partial_statu) #创建一个分布式属性，mesh和partial_status不变，更新dims_mapping。静态图需要搞新的
                
                #Note: tensor_dist_type (type(全局形状,数据类型) , attr_dist_attr(mesh,dims_mapping,partial_status))
                out_type = paddle.base.libpaddle.pir.cvt_to_dist_type(out_type, out_dist_attr) #把原来张量的out_type,转换为 out_dist_type
        """
        tensor_ndim = len(in_value.shape) 
        in_dist_attr = in_value.dist_attr()
        split_mesh_dim = in_dist_attr.dims_mapping[split_axis]
        mesh = in_dist_attr.process_mesh
        # Calculate local shape. In nd_mesh_reshard, multiple tensor axis
        # may be shard and it will call this 1-D s_to_r function on each
        # axis. In this case, we should recompute the local and global shape.
        out_local_shape = list(in_value.shape)
        out_local_shape[split_axis] = int(
            (in_value.shape[split_axis] + mesh.shape[split_mesh_dim] - 1)
            / mesh.shape[split_mesh_dim]
        )
        out_global_shape = list(out_local_shape)
        out_global_shape[0] *= mesh.shape[split_mesh_dim] #这里是否写错了？
        out_type = paddle.pir.create_shaped_type(
            in_value.type(), out_global_shape
        )

        out_dims_mapping = list(in_dist_attr.dims_mapping)
        out_dims_mapping[split_axis] = -1
        out_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            mesh, out_dims_mapping, in_dist_attr.partial_status
        )
        out_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            out_type, out_dist_attr
        )
        return out_type

    #关键中的关键
    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        """
        执行从分片 (sharded) 张量到完全复制 (replicated) 张量的重分配操作。

        Args:
            src_dist_attr : 源张量分布属性,placements和Processmesh
            dst_dist_attr : 这里是 replicated
            src_value : 源张量
            dst_type : 目标张量数据类型和全局形状

        Returns:
            Tensor : dst_value

        Examples:
            假设输入为
            src_dist_attr = (mesh = [0,1,2], out_dims_mapping = [-1,0], partial_status = [-1,-1])

            dst_dist_attr = (mesh = [0,1,2], out_dims_mapping = [-1,-1], partial_status = [-1,-1])

            src_value = [[1,2,3,4,5],
                        [6,7,8,9,10]]

            dst_type = (tensor_type=(形状,数据类型) , tensor_dist_attr = (mesh,dims_mapping,partial_status))
        
            Steps:
            #略过单进程和求split轴


            1. 假设只有一个轴被切分(实际中需要多次调用这个切分),找到第一个被分片的轴,对进程数求余数,非0则需要填充(因为all gather需要吗?)

            2. 如果是均匀的直接调用 reshard_s_to_r_with_padding,否则去最后一个进程去padding(接到step3)

            3.

        """
        if src_dist_attr.process_mesh.size == 1:  #说明只有一个进程(设备)，可以共享张量数据，为什么一个进程也能shard呢？
            dst_value = paddle._C_ops.share_data_(src_value) #就地共享
            #Note: 静态图中，每一个数据都是由一个操作生成的，需要记录下来get_defining_op()，并设置dist_attr
            share_data_op = dst_value.get_defining_op() #获取dst_value的定义op 记作share_data_op，静态图需要记录数据是由什么操作生成的
            # set dist type and dist attr
            dst_value.set_type(dst_type)

            chunk_id = -1 
            if src_value.get_defining_op().dist_attr: #如果源张量有分布式属性
                chunk_id = src_value.get_defining_op().dist_attr.chunk_id #
            #Note: op.dist_attr = (mesh,[src_dist_attr],[dst_dist_attr],chunk_id) 需要操作的输入和输出dist_attr，以及mesh和chunk_id
            share_data_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute( #创建一个 op_dist_attribute
                    src_dist_attr.process_mesh,
                    [src_dist_attr],
                    [dst_dist_attr],
                    chunk_id,
                )
            )
            #Note: 静态图中:
            # src_value.dist_type : (type(形状,数据类型), dist_attr(mesh,dims_mapping,partial_status))
            # 下面是需要设置的 
            # >> defining_op.dist_attr(src_mesh,[src_dist_attr],[dst_dist_attr],chunk_id))  
            # >> dst_value.dist_type : (type(形状,数据类型), dist_attr(mesh,dims_mapping,partial_status))
            return dst_value 

        def get_split_axis_with_dims_mapping(dims_mapping): #找出所有被分片的维度，并记录下来
            split_axis = {}
            for idx, v in enumerate(dims_mapping): #dims_mapping表示张量的维度分布，-1表示没有被分片
                if v != -1:
                    split_axis[idx] = v
            return split_axis  #这里应该打印出来看看

        split_axis_map = get_split_axis_with_dims_mapping( #只记录下来分片长老
            src_dist_attr.dims_mapping
        ) 

        #假设只有一个轴被分片,且只有一维度mesh
        split_axis = -1
        for k, v in split_axis_map.items(): #找到第一个被分片的轴
            split_axis = k
            break
        num_of_process = src_dist_attr.process_mesh.size #进程总数 
        num_of_padding = src_value.shape[split_axis] % num_of_process #求余数得到填充数量 
        is_balanced_split = num_of_padding == 0 #注意这是个判断
        
        #如果是不需要padding的，直接调用reshard_s_to_r_padding就可以了
        if is_balanced_split:
            new_value = self.reshard_s_to_r_with_padding(
                src_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                num_of_padding,
            )
            return new_value #这里就return了

        else: #这里就是关键，我要在别的也实现这个
            # find the last one，避免影响其他进程的数据分布(这里还不太确定为什么？)
            need_padding = (
                paddle.distributed.get_rank() #获取当前进程的rank
                == src_dist_attr.process_mesh.process_ids[-1] #找到mesh的最后一个 proces_ids来 进行padding
            )

            # get padding_num
            avg_size_on_split_axis = int( #求解每个设备上多少个元素
                (src_value.shape[split_axis] + num_of_process - 1)
                / num_of_process
            )
            padding_num = ( 
                avg_size_on_split_axis * num_of_process
                - src_value.shape[split_axis]
            )

            if need_padding:
                # set right _local_shape
                local_shape_at_split_axis = src_value.shape[
                    split_axis
                ] - avg_size_on_split_axis * (num_of_process - 1) #计算最后一个进程 上的 本地形状,比如5-2*2 = 1
                local_shape = src_value._local_shape #获取原来的列表用来下面更新
                local_shape[split_axis] = local_shape_at_split_axis #更新被分割维度的 l·ocal_shape 
                #难道是因为只有local化才能进行concat吗？
                tmp_src_type = paddle.base.libpaddle.pir.cvt_to_dist_type( #将张量从本地类型转换为分布式类型（本身不是分布式类型吗shard）
                    src_value.type(), src_dist_attr, list(local_shape)
                )
                src_value.set_type(tmp_src_type) #这里更新了 src的类型，为什么要调整？
                #创建填充张量
                padding_shape = src_value._local_shape #这个也需要打印出来看一下
                padding_shape[split_axis] = padding_num
                print(f"padding shape is:{padding_shape}")
                padding_tensor = paddle.full( 
                    padding_shape,
                    0.0,
                    src_value.dtype,
                )
                tmp_src_type1 = paddle.base.libpaddle.pir.cvt_to_dist_type( #得到填充张量的分布式类型
                    padding_tensor.type(), dst_dist_attr
                )
                padding_tensor.set_type(tmp_src_type1) #设置填充张量类型
                padding_tensor.get_defining_op().dist_attr = ( #创建并分配分布属性
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        dst_dist_attr.process_mesh, [], [dst_dist_attr]
                    )
                )
                #将原张量和填充张量沿着切分轴 拼接起来
                concat_value = paddle._C_ops.concat(
                    [src_value, padding_tensor], split_axis
                )
                # set concat dist_attr
                axis_dist_attr = ( #将拼接起来的轴，设置为未被分片 -1
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        src_dist_attr.process_mesh, [-1], {}
                    )
                )
                concat_value.get_defining_op().dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute( #pir创建 op的分布属性
                        src_dist_attr.process_mesh, #src_dist_attr的mesh
                        [ 
                            paddle.base.libpaddle.pir.create_array_attribute( #pir创建 矩阵属性
                                [src_dist_attr, dst_dist_attr]  #为什么会有目标分布属性？
                            ),
                            axis_dist_attr,
                        ],
                        [src_dist_attr],  #放到src_dist_attr里
                    )
                )
                # set concat_value type
                concat_global_shape = list(src_value.shape) #获取concat后的全局类型
                concat_global_shape[split_axis] = ( #设置分割后的切分维度尺寸为 2*3
                    avg_size_on_split_axis * num_of_process
                )
                concat_type = paddle.pir.create_shaped_type( #创建形状type
                    src_value.type(), concat_global_shape
                )
                concat_type = paddle.base.libpaddle.pir.cvt_to_dist_type(  #创建分布式type，这些type都是什么？
                    concat_type, src_dist_attr
                )
                concat_value.set_type(concat_type) #更新type
                #执行重分片
                new_value = self.reshard_s_to_r_with_padding(
                    concat_value,
                    split_axis,
                    src_dist_attr,
                    dst_dist_attr,
                    dst_type,
                    padding_num,
                )
                return new_value
            else: #其他rank
                new_value = self.reshard_s_to_r_with_padding(
                    src_value,
                    split_axis,
                    src_dist_attr,
                    dst_dist_attr,
                    dst_type,
                    padding_num,
                )
                return new_value

    #reshard实际操作，可以先讨论这里！！！
    def reshard_s_to_r_with_padding(
        self,
        src_value, #原始张量
        split_axis, #切分的维度，例如 0
        src_dist_attr, #原张量的分布属性，例如mesh
        dst_dist_attr,
        dst_type, #数据类型
        padding_num=0, 
    ):
        """
        Args:
            src_value : 源张量
            split_axis : 切分轴 
            src_dist_attr : 源张量分布属性,placements和Processmesh
            dst_dist_attr : 这里是 replicated
            dst_type : 目标张量数据类型和全局形状
            padding_num : 填充数量

        Returns:
            Tensor : dst_value

        Examples1:
            假设输入为

            Note:这里src_value已经填充过了
            src_value = [[1,2,3,4,5,0], 
                        [6,7,8,9,10,0]] 

            src_dist_attr = (mesh = [0,1,2], out_dims_mapping = [-1,0], partial_status = [-1,-1])

            dst_dist_attr = (mesh = [0,1,2], out_dims_mapping = [-1,-1], partial_status = [-1,-1])

            dst_type = (tensor_type=(形状,数据类型) , tensor_dist_attr = (mesh,dims_mapping,partial_status))

            padding_num = 1

            Steps:
            1. 获取源张量信息
            得到src的 mesh和process_num,获取源张量 defining op的chunk_id
                src_mesh = [0,1,2]
                num_of_process = 3
                chunk_id = ???

            2. 进程间通信
            创建进程组,all_gather得到all_gather_value,推测allgather_type并设置
                group = ([1,2,3])

                #Note:这里是global tensor,local tensor是[[1,2,3,4,5,0], [6,7,8,9,10,0]]
                all_gather_value = [
                                    [[1,2,3,4,5,0], 
                                    [6,7,8,9,10,0]],

                                    [[1,2,3,4,5,0], 
                                    [6,7,8,9,10,0]],

                                    [[1,2,3,4,5,0], 
                                    [6,7,8,9,10,0]]] 
                
                allgather_type = tensor_dist_type (type(全局形状,数据类型) , tensor_dist_attr(mesh,dims_mapping,partial_status))
                               = dist_type(type((3,2,5) , float)  , dist_attr([0,1,2],[-1,-1,-1],[-1,-1,-1]))
                allgather.set_type()

            3. 创建新的张量分布属性 
                new_dist_attr = tensor_dist_attr(mesh,dims_mapping,partial_status)
                              = ([0,1,2],[-1,-1,-1],[-1,-1,-1])

            4. 设置allgather_value的defining_op的dist_attr
                allgather_value.defining_op.dist_attr = defining_op.dist_attr(src_mesh,[src_dist_attr],[dst_dist_attr],chunk_id))
                                                      = ([0,1,2],[src_dist_attr],[new_dist_attr],0/1/2)

                #Note:这里看到的究竟是local还是global
                #Note: 静态图搞一个新张量,需要创建并设置:
                                    defining_op的dist_attr
                                    tensor的dist_attr
                                    tensor的dist_type(type,dist_attr)
            
            5. 处理切分轴,获取allgather_op,对其result进行split得到split_values,
               获取第一个分割结果的builtin_difining_op,
               在得到其operand_source(0).difining_op()作为pd_splite_op？

                allgather_op = 
                split_values = paddle._C_ops.split_with_num(allgather_op.result(0), num_proc,0)
                builtin_split_op = split_values[0].get_defining_op()

                #Note: 这个不知道为什么要搞出来pd_split_op,普通的split不够吗？
                pd_splite_op = builtin_split_op.operand_source(0).get_defining_op()
                pd_splite_op.dist_attr = copy_op_attr_with_new_member(pd_splite_op.dist_attr, new_chunk_id=chunk_id )
            
            6. 修正pd_split_op的输出类型


            7. 处理填充部分


            8. 处理不需要填充的情况

        """

        src_mesh = src_dist_attr.process_mesh
        num_of_process = len(src_mesh.process_ids)
        #获取chunk_id,分块张量的id
        chunk_id = -1
        if src_value.get_defining_op().dist_attr: # 如果源张量的定义操作有分布属性,可能有 mesh,dims mapping,partial status
            chunk_id = src_value.get_defining_op().dist_attr.chunk_id
        
        #开始进程通信
        group = new_process_group(sorted(src_mesh.process_ids)) #创建进程组，允许进程通信（包含全部）
        allgather_value = paddle._C_ops.all_gather(
            src_value, group.id, num_of_process
        ) #形成一个更大张量，例如[[1,2,3,4,5],[6,7,8,9,10]] -> [[[1,2,3,4,5],[6,7,8,9,10]],[[1,2,3,4,5],[6,7,8,9,10]],[[1,2,3,4,5],[6,7,8,9,10]]]
        allgather_type = self.infer_allgather_dist_type(src_value, split_axis) #推测分布类型
        allgather_value.set_type(allgather_type) #设置分布类型

        # set op_dist_attr 
        #进一步，新建一个分布属性
        new_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh, #目标进程的网格，这个是提前输入的
            [-1] * len(dst_dist_attr.dims_mapping), #全部都是 -1了，转变为replicated了
            dst_dist_attr.partial_status, #这个也是提前设置的
        )
        #设置操作的分布属性，这是什么概念？
        #保证计算图中的操作行为与张量分布相匹配,这是为了静态图而专门设置的吗？
        allgather_value.get_defining_op().dist_attr = ( #设置all_gather的操作分布属性
            paddle.base.libpaddle.pir.create_op_dist_attribute( #创建op_dist_attribute
                src_mesh, # 源进程网格
                [src_dist_attr], # 输入的分布属性列表
                [new_dist_attr], # 输出的分布属性列表
                chunk_id # 分块 ID
            )
        )
        #非0分片轴，这是因为本来allgather之后，就是需要分开的，除非刚好分片轴就是 0
        if split_axis != 0 or padding_num != 0: # 如果分片轴不是 0 或者存在填充
            # 获取 all_gather 操作的定义操作
            allgather_op = allgather_value.get_defining_op()
            # 对 all_gather 的结果在第 0 轴上按进程数量进行分割
            split_values = paddle._C_ops.split_with_num(
                allgather_op.result(0), 
                num_of_process, 
                0
            )
            builtin_split_op = split_values[0].get_defining_op()# 获取内置的 split 操作（第一个分割结果的定义操作）
            pd_splite_op = builtin_split_op.operand_source(0).get_defining_op()# 获取 Paddle 定义的 split 操作
            pd_splite_op.dist_attr = copy_op_attr_with_new_member( 
                pd_splite_op.dist_attr, new_chunk_id=chunk_id  # 更新 pd_splite_op 的分布属性，设置新的 chunk_id
            )

            # fix the split_with_num dist attribtue.
            #修正 split_with_num 操作的分布属性和类型
            new_inner_types = []
            for sub_value in split_values:
                #每个子张量的类型转换为与 allgather_value 相同的分布类型
                new_inner_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    sub_value.type(), allgather_value.dist_attr()
                )
                new_inner_types.append(new_inner_type) #添加进去所有的new_inner_type
                sub_value.set_type(new_inner_type) #设置type
            # 创建包含所有子张量类型的向量类型
            vec_type = paddle.base.libpaddle.pir.create_vec_type( 
                new_inner_types
            )
            #更新 split_with_num 操作的输出类型
            pd_splite_op.result(0).set_type(vec_type)
            #填充过的张量需要处理，去除填充部分
            if padding_num != 0:
                # 对最后一个子张量再次进行分割，去除填充部分
                tmp_split_values = paddle._C_ops.split(
                    split_values[-1],
                    [
                        split_values[-1].shape[split_axis] - padding_num,
                        padding_num,
                    ],
                    split_axis,
                )
                # 获取新的 split 操作的定义操作 并更新
                split_op = tmp_split_values.get_defining_op() #获取 
                split_op.dist_attr = copy_op_attr_with_new_member( # 更新 split_op 的分布属性
                    split_op.dist_attr, new_chunk_id=chunk_id
                )
                # 更新最后一个子张量为去除填充后的张量
                split_values[-1] = tmp_split_values[0]
                # 将所有子张量在指定轴上进行拼接
                concat_value = paddle._C_ops.concat(split_values, split_axis)
                # 获取 concat 操作的定义操作并更新
                concat_op = concat_value.get_defining_op()
                concat_op.dist_attr = copy_op_attr_with_new_member(# 更新 concat_op 的分布属性
                    concat_op.dist_attr, new_chunk_id=chunk_id
                )
                return concat_value #终于可以返回了
            else: #没有填充的话
                concat_value = paddle._C_ops.concat(split_values, split_axis)
                # fold builtin.split op and builtin.combine op 
                # 获取 concat 操作的定义操作并更新操作的分布属性
                concat_op = concat_value.get_defining_op()
                concat_op.dist_attr = copy_op_attr_with_new_member(
                    concat_op.dist_attr, new_chunk_id=chunk_id
                )
                # 获取内置的 combine 操作（用于合并张量）
                builtin_combine_op = concat_op.operand_source(
                    0
                ).get_defining_op()
                concat_op.operand(0).set_source(pd_splite_op.result(0))#更新 concat_op 的输入源为 pd_splite_op 的输出
                # 删除内置的 combine 和 split 操作，优化计算图
                builtin_combine_op.erase()
                builtin_split_op.erase()
                return concat_value

        return allgather_value

# 跨ProcessMesh,转换了不同的mesh
class SToRReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if (
            in_mesh.ndim != 1
            or out_mesh.ndim != 1
            or in_mesh.shape != out_mesh.shape
        ):
            return False

        if in_mesh == out_mesh:
            return False

        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        #处理分片张量在两个不同 ProcessMesh 之间的迁移，确保张量保持同样的分布状态。
        same_status_func = SameStatusReshardFunction() 
        #目标 ProcessMesh 上创建一个临时的分布属性 tmp_dist_attr，
        # 但使用源张量的分片映射（dims_mapping）和部分计算状态（partial_status）
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            src_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        #src类型转换为tmp
        tmp_dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            src_value.type(), tmp_dist_attr
        )
        #进行同状态的reshard，张量从源 ProcessMesh 迁移到目标 ProcessMesh
        out_value = same_status_func.reshard( #用这个方法来进行reshard
            src_dist_attr, tmp_dist_attr, src_value, tmp_dst_type
        )

        #完成从分片到复制的重分片
        s_to_r_func = SToRReshardFunction() 
        #检测是否可行
        assert s_to_r_func.is_suitable(
            tmp_dist_attr, dst_dist_attr
        ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
        #实际调用并返回
        return s_to_r_func.reshard(
            tmp_dist_attr, dst_dist_attr, out_value, dst_type
        )
