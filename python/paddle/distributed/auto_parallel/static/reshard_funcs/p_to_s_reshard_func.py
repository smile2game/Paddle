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
        import ipdb
        # if dist.get_rank() == 0:
        #     ipdb.set_trace()

        #检测 partial_status
        src_reduce_type = src_dist_attr.partial_status[0] #铁sum
        assert (
            src_reduce_type == paddle.base.core.ReduceType.kRedSum
        ), f"The p to s reshard func only support sum op, but received {src_reduce_type}"
        split_axis = dst_dist_attr.dims_mapping.index(0) 
        if split_axis != 0:
            perm = list(range(0, len(src_value.shape)))
            perm[0] = split_axis
            perm[split_axis] = 0
            src_value = paddle._C_ops.transpose(src_value, perm)
            tmp_dims_mapping = dst_dist_attr.dims_mapping
            tmp_dims_mapping[split_axis] = -1
            tmp_dims_mapping[0] = 0
            dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_dims_mapping=tmp_dims_mapping
            )

            global_dst_attr = dst_type.as_dist_type().dist_attr()
            global_dims_mapping = global_dst_attr.dims_mapping
            axis = global_dims_mapping[0]
            global_dims_mapping[0] = global_dims_mapping[split_axis]
            global_dims_mapping[split_axis] = axis
            global_dist_attr = copy_dist_attr_with_new_member(
                global_dst_attr, new_dims_mapping=global_dims_mapping
            )
            dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), global_dist_attr
            )
        
        #检测是否需要padding padding
        num_of_process = len(src_dist_attr.process_mesh.process_ids)
        remainder_of_padding = src_value.shape[split_axis] % num_of_process #求余数
        is_balanced_split = remainder_of_padding == 0

        #如果不需要padding
        if is_balanced_split:
            new_value = self.reshard_p_to_s_with_padding(
                src_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                # padding_num,
                )
            return new_value
        else:
            print("先完成padding吧,全都转置了,默认在0维度")
            out_split_axis = 0 # out_split_axis = GetSplitAxisWithDimsMapping(out_dist_attr.dims_mapping()).begin()->first;
            #所有rank都需要 padding的
            avg_size_on_split_axis = int((src_value.shape[out_split_axis] + num_of_process -1) / num_of_process) #avg = 3
            padding_num = avg_size_on_split_axis * num_of_process -src_value.shape[split_axis] #padding_num = 3*2 - 5
            #我不明白这个 set_type的作用,这里是在更新局部形状吗?那我是不是不需要
            # tmp_src_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            #         src_value.type(), src_dist_attr, list(local_shape)
            #     )
            # src_value.set_type(tmp_src_type)

            #创建padding张量
            padding_shape = src_value._local_shape
            padding_shape[out_split_axis] = padding_num
            #使用 pd_op.full来创建
            padding_tensor = paddle.full(   
                padding_shape,
                0.0,
                src_value.dtype,
            ) 
            #设置op.full.dist_attr
            padding_tensor.get_defining_op().dist_attr = (paddle.base.libpaddle.pir.create_op_dist_attribute
                (
                    dst_dist_attr.process_mesh, [], [dst_dist_attr]
                ))

            #pd_op.concat拼接起来src_tensor和padding,有很多dist_attr要设置,怎么知道要设置哪些,以及如何设置?
            concat_value = paddle._C_ops.concat([src_value, padding_tensor], split_axis)
            #用来设置 concat op的dist_attr
            axis_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                src_dist_attr.process_mesh, [-1], {} #这里是否需要设置一下 partial
                )
            )
            #为什么这样设置,可以去哪里查一下?
            concat_value.get_defining_op().dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_dist_attr.process_mesh,
                    [
                        paddle.base.libpaddle.pir.create_array_attribute(
                                [src_dist_attr, dst_dist_attr]
                        ),
                        axis_dist_attr,
                    ],
                    [src_dist_attr]
                )
            )

            #设置concat_value type
            new_value = self.reshard_p_to_s_with_padding(
                concat_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                padding_num,
                )

            return new_value

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
            if split_axis != 0:
                dst_value = paddle._C_ops.transpose(dst_value, perm) #转置
            return dst_value