import paddle
from paddle.distributed.utils.stream_utils import ExecutionStreamType

from ..process_group import new_process_group #创建进程组
from .base_reshard_func import (
    ReshardFunction,
    copy_dist_attr_with_new_member,
    is_partial,
    is_shard,
)
import paddle.distributed as dist
import ipdb


class PToSReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
            return False

        if not is_shard(dst_dist_attr):
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

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_reduce_type = src_dist_attr.partial_status[0] #检测 partial_status是sum
        assert (
            src_reduce_type == paddle.base.core.ReduceType.kRedSum
        ), f"The p to s reshard func only support sum op, but received {src_reduce_type}"
        split_axis = dst_dist_attr.dims_mapping.index(0) 
        #转置重排
        print(f"split_axis is {split_axis}")
        permulate = False
        if split_axis != 0:
            perm = list(range(0, len(src_value.shape)))
            print(f"perm is {perm}")
            perm[0] = split_axis
            perm[split_axis] = 0
            src_value = paddle._C_ops.transpose(src_value, perm) #交换 0 和 split_axis这个矩阵
            tmp_dims_mapping = dst_dist_attr.dims_mapping
            tmp_dims_mapping[split_axis] = -1
            tmp_dims_mapping[0] = 0 
            dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_dims_mapping=tmp_dims_mapping
            ) #调整dst_dist_attr的dims_mapping，交换了0和split_axis

            global_dst_attr = dst_type.as_dist_type().dist_attr() #从dst_type中抽取出来dst_dist_attr
            print(f"dst_type is {dst_type}")
            print(f"global_dst_attr is {global_dist_attr}") 
            global_dims_mapping = global_dst_attr.dims_mapping
            axis = global_dims_mapping[0]
            global_dims_mapping[0] = global_dims_mapping[split_axis] 
            global_dims_mapping[split_axis] = axis #交换global_dims_mapping的0和split_axis
            global_dist_attr = copy_dist_attr_with_new_member(
                global_dst_attr, new_dims_mapping=global_dims_mapping 
            ) #前面dst_dist_attr调整过了,global_dst_attr还需要重新调整一遍,为什么???
            dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), global_dist_attr
            ) #这里是什么意思???
            print(f"dst_type is {dst_type}")
            permulate = True
            split_axis = 0

        #检测是否需要padding padding
        num_of_process = len(src_dist_attr.process_mesh.process_ids)
        remainder_of_padding = src_value.shape[split_axis] % num_of_process #求余数
        is_balanced_split = remainder_of_padding == 0

        #如果不需要padding
        if is_balanced_split:
            print(f"balanced,no need to padding!")
            dst_value = self.reshard_p_to_s_with_padding(
                src_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                # padding_num,
                )
            if permulate:
                dst_value = paddle._C_ops.transpose(dst_value, perm) #转置
            return dst_value
        else:
            """
            padding steps:
                tensor_shape: 3 x 2
                mesh = [0,1]
                shard = 0
                padding_shape: 1 x 2
                concat_value: 4 x 2
                shard_value: 2x2 - 2x2
                split_value(last rank): 1x2 
            """
            print("unbalanced,padding->conact->reduce_scatter->split,after permulate,on all split on dim0")
            #所有rank都需要 padding的
            avg_size_on_split_axis = int((src_value.shape[split_axis] + num_of_process -1) / num_of_process) #avg = 3
            padding_num = avg_size_on_split_axis * num_of_process -src_value.shape[split_axis] #padding_num = 3*2 - 5
            #不明白这个 set_type的作用,这里是在更新局部形状吗?那我是不是不需要
            # tmp_src_type = paddle.base.libpaddle.pir.cvt_to_dist_type(src_value.type(), src_dist_attr, list(src_value._local_shape))
            # print(f"before set src_value.type is {src_value.type}")
            # src_value.set_type(tmp_src_type)
            print(f"src_value.type is {src_value.type}")
            # print(f"\nand its dist_attr is {src_value.dist_attr},\ntmp_src_type is {tmp_src_type}")

            #######################################使用 pd_op.full来创建padding_tensor###########################
            padding_shape = src_value._local_shape 
            padding_shape[split_axis] = padding_num
            print(f"every rank has {src_value._local_shape}(3x2),and padding {padding_num}(1x2)")
            print(f"rank{dist.get_rank()} padding_shape is {padding_shape}")
            padding_tensor = paddle.full(   
                padding_shape,
                0.0,
                src_value.dtype,
            ) 
            # help(paddle.base.libpaddle.pir.cvt_to_dist_type)
            tmp_src_type = paddle.base.libpaddle.pir.cvt_to_dist_type( #设置分布式属性,直接 = src_dist_attr不行,是read-only的
                padding_tensor.type(),src_dist_attr
            )
            padding_tensor.set_type(tmp_src_type) #修改一个tensor的dist_attr要通过type来修改

            print(f"padding_tensor dist_attr is {padding_tensor.dist_attr}")
            # help(paddle.base.libpaddle.pir.create_op_dist_attribute)
            print(f"\nbefore set,padding_tensor.get_defining_op().dist_attr is {padding_tensor.get_defining_op().dist_attr}")
            padding_tensor.get_defining_op().dist_attr = ( #设置 pd_op.full.dist_attr
                paddle.base.libpaddle.pir.create_op_dist_attribute( 
                    src_dist_attr.process_mesh,  #都是一样的
                    [], 
                    [src_dist_attr] #
                ))
            print(f"after set,padding_tensor.get_defining_op().dist_attr is {padding_tensor.get_defining_op().dist_attr}")
            
            #######################################使用 pd_op.concat来拼接src_value + padding_tensor###########################
            # help(paddle._C_ops.concat)
            print(f"\nsrc_value.dist_attr is {src_value.dist_attr},\npadding_tensor.dist_attr is {padding_tensor.dist_attr}")
            concat_value = paddle._C_ops.concat(
                [
                    src_value,
                    padding_tensor
                ], 
                split_axis)
            print(f"concat_value.dist_attr is {concat_value.dist_attr}")
            #用来设置 concat op的dist_attr
            axis_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                src_dist_attr.process_mesh, [-1], {} #这里是否需要设置一下 partial
                )
            )
            print(f"\nbefore set(no need to set),concat_value.get_defining_op().dist_attr is {concat_value.get_defining_op().dist_attr}")
            # concat_value.get_defining_op().dist_attr = (
            #     paddle.base.libpaddle.pir.create_op_dist_attribute(
            #         src_dist_attr.process_mesh,
            #         [
            #             paddle.base.libpaddle.pir.create_array_attribute(
            #                     [src_dist_attr, dst_dist_attr]
            #             ), #operant0.dist_attr
            #             axis_dist_attr, #operant1.dist_attr
            #         ], #输入的操作数
            #         [src_dist_attr] #输出的操作数
            #     )
            # )
            # print(f"after set,concat_value.get_defining_op().dist_attr is {concat_value.get_defining_op().dist_attr}")

            #设置concat_value type
            dst_value = self.reshard_p_to_s_with_padding(
                concat_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                padding_num,
                )

            if permulate:
                dst_value = paddle._C_ops.transpose(dst_value, perm) #转置恢复
                #dims_mapping不需要修正吗?
            return dst_value

    def reshard_p_to_s_with_padding(
        self,
        src_value,
        split_axis,
        src_dist_attr,
        dst_dist_attr,
        dst_type, #我想知道这个type是什么?
        padding_num = 0,
        ):
            group = new_process_group(sorted(src_dist_attr.process_mesh.process_ids)) #创建进程组
            #处理输出值
            #1 reduce_scatter得到边
            dst_value = paddle._C_ops.reduce_scatter( 
                src_value, group.id, len(src_dist_attr.process_mesh.process_ids)
            )
            #2 设置执行流 为默认
            dst_value.get_defining_op().set_execution_stream(
                ExecutionStreamType.DefaultStream.value
            )
            #3 set dist type and dist attr
            #3.1 设置值 的 type
            dst_value.set_type(dst_type)
            #3.2 设置op 的dist_attr
            dst_value.get_defining_op().dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_dist_attr.process_mesh, 
                    [src_dist_attr], 
                    [dst_dist_attr], 
                    src_value.get_defining_op().dist_attr.chunk_id
                )
            )

            #还需要切除掉最后一个 rank上,被padding到的部分
            if padding_num!=0:
                if dist.get_rank() == dst_dist_attr.process_mesh.process_ids[-1]:
                    dst_value = paddle._C_ops.split(
                        dst_value,
                        [
                            dst_value.shape[split_axis]-padding_num,
                            padding_num,
                        ],
                        split_axis,
                    )[0]
                print(f"dst_value is {dst_value},its dist_attr is {dst_value.dist_attr}")
            return dst_value