# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.auto_parallel.static.pir_pass import (
    # apply_reshard_pass,
    ReshardPasses
)
import ipdb

#partial -> shard
class TestReshardPToS:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        # self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._mesh = dist.ProcessMesh([0,1], dim_names=["x"])
        # self._out_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])
        self.rank = dist.get_rank()

    def run_pir_test_case(self):
        paddle.enable_static()
        if self._backend == "gpu": #只测 GPU
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2 #设置参数
        SEQ_LEN = 2
        HIDDEN_SIZE = 6
        MP_SIZE = 2 

        #编写静态图
        with paddle.pir_utils.IrGuard(): #进入 pir模式 
            main_program = paddle.base.Program() 
            with paddle.base.program_guard(main_program): #指定默认Program
                #创建一个partial张量，sum的
                w0 = paddle.pir.core.create_parameter( #创建这个张量
                    dtype = "float32",
                    shape = [SEQ_LEN,HIDDEN_SIZE],
                    name = "w0",
                    initializer = paddle.nn.initializer.Uniform(),
                )
                #创建一个partial张量,partial status
                input_tensor = dist.shard_tensor(w0,self._mesh,[dist.Partial()])
                reshard_tensor = paddle._C_ops.reshard(
                                                        input_tensor,
                                                        self._mesh,
                                                        [dist.Shard(self._shard)], #
                                                    )
            ReshardPasses.apply_reshard_pass(main_program)
        
        #研究属性设置
        ops = [op.name() for op in main_program.global_block().ops]

        if self._shard == 0:
            np.testing.assert_equal(main_program.num_ops(),3) #让我来猜一下数量
            std_ops = [
                "builtin.parameter",
                "dist_op.shard_tensor",
                # "pd_op.transpose", #等于0就不用转置了
                "pd_op.reduce_scatter",
                # "pd_op.transpose"
            ]
            np.testing.assert_equal(
                ops,
                std_ops,
            )
            print("op数量和名称匹配!")


        if self._shard == 1:
            np.testing.assert_equal(main_program.num_ops(),5) #让我来猜一下数量
            std_ops = [
                "builtin.parameter",
                "dist_op.shard_tensor",
                "pd_op.transpose", #等于0就不用转置了
                "pd_op.reduce_scatter",
                "pd_op.transpose"
            ]
            np.testing.assert_equal(
                ops,
                std_ops,
            )
        
            #开始测试属性
        for op in main_program.global_block().ops:
            if op.name() == "pd_op.reduce_scatter":
                #check op.dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh #只用到了一个mesh
                print("reduce_scatter_op.dist_attr匹配")
                #check op_operand
                op_operand_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr() #得到operand的分布式属性
                assert op_operand_dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.dims_mapping == [-1,-1]
                assert op_operand_dist_attr.partial_status == {0: paddle.base.core.ReduceType.kRedSum} #怎么创建一个reduce type???
                print("reduce_scatter_op.operand(0) dist_attr匹配")
                #check op_result
                op_result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr() 
                assert op_result_dist_attr.process_mesh == self._mesh
                if self._shard == 0:
                    assert op_result_dist_attr.dims_mapping == [0, -1]
                else:
                    assert op_result_dist_attr.dims_mapping == [-1, 0]
                assert op_result_dist_attr.partial_status == {}
                print("reduce_scatter_op.result(0) dist_attr匹配")
                #check op_value.dist_attr
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type() #这里不满足,因为他是shard的
                assert op_value.dist_attr().process_mesh == self._mesh
                if self._shard == 0:
                    assert op_value.dist_attr().dims_mapping == [0, -1]
                else:
                    assert op_result_dist_attr.dims_mapping == [-1, 0]
                assert op_value.dist_attr().partial_status == {}
                print("reduce_scatter_op.value.dist_attr匹配")
        


    def run_pir_unbalanced_split_test_case(self):
        paddle.enable_static()
        if self._backend == "gpu": #只测 GPU
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2 #设置参数
        SEQ_LEN = 3
        HIDDEN_SIZE = 7
        MP_SIZE = 2 

        #编写静态图
        with paddle.pir_utils.IrGuard(): #进入 pir模式 
            main_program = paddle.base.Program() 
            with paddle.base.program_guard(main_program): #指定默认Program
                #创建一个partial张量，sum的
                w1 = paddle.pir.core.create_parameter( #创建这个张量
                    dtype = "float32",
                    shape = [SEQ_LEN,HIDDEN_SIZE],
                    name = "w1",
                    initializer = paddle.nn.initializer.Uniform(),
                )
                #创建一个partial张量,partial status
                input_tensor1 = dist.shard_tensor(w1,self._mesh,[dist.Partial()])
                reshard_tensor1 = paddle._C_ops.reshard(
                                                        input_tensor1,
                                                        self._mesh,
                                                        [dist.Shard(self._shard)], #
                                                    )
            ReshardPasses.apply_reshard_pass(main_program)
        
        #研究属性设置
        ops = [op.name() for op in main_program.global_block().ops]
        print(ops)

        if self._shard == 0:
            if self.rank != self._mesh.process_ids[-1]:
                np.testing.assert_equal(main_program.num_ops(),7) #让我来猜一下数量
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    # "pd_op.transpose", #等于0就不用转置了
                    "pd_op.full",
                    "pd_op.full", #为什么多一个?
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",
                    # "pd_op.transpose"
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )
            else:
                np.testing.assert_equal(main_program.num_ops(),11) #让我来猜一下数量
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    # "pd_op.transpose", #等于0就不用转置了
                    "pd_op.full",
                    "pd_op.full", #为什么多一个?
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",

                    'pd_op.full_int_array', #这是末尾多出来的
                    'pd_op.full', 
                    'pd_op.split', 
                    'builtin.split'
                    # "pd_op.transpose"
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )
            print(f"rank{self.rank}的op数量和名称匹配!")


        if self._shard == 1:
            if self.rank != self._mesh.process_ids[-1]:
                np.testing.assert_equal(main_program.num_ops(),9) #让我来猜一下数量
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    "pd_op.transpose", #等于0就不用转置了
                    "pd_op.full",
                    "pd_op.full", #为什么多一个?
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",
                    "pd_op.transpose"
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )
            else:
                np.testing.assert_equal(main_program.num_ops(),13) #让我来猜一下数量
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    "pd_op.transpose", #等于0就不用转置了
                    "pd_op.full",
                    "pd_op.full", #为什么多一个?
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",

                    'pd_op.full_int_array', #这是末尾多出来的
                    'pd_op.full', 
                    'pd_op.split', 
                    'builtin.split',
                    "pd_op.transpose"
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )
            print(f"rank{self.rank}的op数量和名称匹配!")
        
        for op in main_program.global_block().ops:
            if op.name() == "pd_op.reduce_scatter": #所有 rank
                #check op.dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh #只用到了一个mesh
                print("reduce_scatter_op.dist_attr匹配")
                #check op_operand,取不出来吗?
                op_operand_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr() #得到operand的分布式属性
                assert op_operand_dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.dims_mapping == [-1,-1]
                assert op_operand_dist_attr.partial_status == {0: paddle.base.core.ReduceType.kRedSum} #怎么创建一个reduce type???
                print("reduce_scatter_op.operand(0) dist_attr匹配")
                #check op_result
                op_result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr() 
                assert op_result_dist_attr.process_mesh == self._mesh
                if self._shard == 0:
                    assert op_result_dist_attr.dims_mapping == [0, -1]
                else:
                    assert op_result_dist_attr.dims_mapping == [-1, 0]
                assert op_result_dist_attr.partial_status == {}
                print("reduce_scatter_op.result(0) dist_attr匹配")
                #check op_value.dist_attr
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type() #这里不满足,因为他是shard的
                assert op_value.dist_attr().process_mesh == self._mesh
                if self._shard == 0:
                    assert op_value.dist_attr().dims_mapping == [0, -1]
                else:
                    assert op_result_dist_attr.dims_mapping == [-1, 0]
                assert op_value.dist_attr().partial_status == {}
                print("reduce_scatter_op.value.dist_attr匹配")
            elif op.name() == 'pd_op.concat': 
                #check op.dist_attr
                assert op.dist_attr.num_operands() == 2
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh #只用到了一个mesh
                print("concat_op.dist_attr匹配")
                #check op_operand
                operand_1_dist_attrs = op.dist_attr.operand(0).as_array_attr() #得到operand的分布式属性
                assert len(operand_1_dist_attrs) == 2
                operand_1_dist_attr_1 = operand_1_dist_attrs[0].as_tensor_dist_attr()
                operand_1_dist_attr_2 = operand_1_dist_attrs[1].as_tensor_dist_attr()
                assert operand_1_dist_attr_1.process_mesh == self._mesh
                assert operand_1_dist_attr_1.dims_mapping == [-1, -1]
                assert operand_1_dist_attr_1.partial_status == {0: paddle.base.core.ReduceType.kRedSum}
                assert operand_1_dist_attr_2.process_mesh == self._mesh
                assert operand_1_dist_attr_2.dims_mapping == [-1, -1]
                assert operand_1_dist_attr_2.partial_status == {0: paddle.base.core.ReduceType.kRedSum}
                print("concat_op.operand(0) dist_attr匹配")
                op_operand1_dist_attr = op.dist_attr.operand(1).as_tensor_dist_attr()
                print("concat_op.operand(1) dist_attr匹配")

                #check op_result
                op_result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr() 
                assert op_result_dist_attr.process_mesh == self._mesh
                assert op_result_dist_attr.dims_mapping == [-1, -1]
                assert op_result_dist_attr.partial_status == {0: paddle.base.core.ReduceType.kRedSum}
                print("concat_op.result(0) dist_attr匹配")
                #check op_value.dist_attr
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type() #这里不满足,因为他是shard的
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [-1, -1]
                assert op_value.dist_attr().partial_status == {0: paddle.base.core.ReduceType.kRedSum}
                print("concat_op.value.dist_attr匹配")
            elif op.name() == 'pd_op.split': #只有 最后一个 rank
                if self.rank == self._mesh.process_ids[-1]:
                    #check op.dist_attr
                    print(f"op.dist_attr.num_operands() is {op.dist_attr.num_operands()}")
                    assert op.dist_attr.num_operands() == 3
                    assert op.dist_attr.num_results() == 1
                    assert op.dist_attr.process_mesh == self._mesh #只用到了一个mesh
                    print("split_op.dist_attr匹配")

                    #check op_operand
                    op_operand_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr() #得到operand的分布式属性
                    assert op_operand_dist_attr.process_mesh == self._mesh
                    if self._shard == 0:
                        print(f"op_operand_dist_attr.dims_mapping is {op_operand_dist_attr.dims_mapping}")
                        assert op_operand_dist_attr.dims_mapping == [0, -1]
                    else:
                        assert op_operand_dist_attr.dims_mapping == [-1, 0]
                    assert op_operand.dist_attr().partial_status == {}
                    print("split_op.operand(0) dist_attr匹配")
                    # check op_result
                    op_result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr() 
                    assert op_result_dist_attr.process_mesh == self._mesh
                    if self._shard == 0:
                        assert op_result_dist_attr.dims_mapping == [0, -1]
                    else:
                        assert op_result_dist_attr.dims_mapping == [-1, 0]
                    assert op_result.dist_attr().partial_status == {}
                    print("split_op.result(0) dist_attr匹配")
                    # check op_value.dist_attr
                    op_value = op.result(0)
                    assert op_value.is_dense_tensor_type()
                    assert op_value.is_dist_dense_tensor_type() #这里不满足,因为他是shard的
                    assert op_value.dist_attr().process_mesh == self._mesh
                    if self._shard == 0:
                        assert op_value.dist_attr().dims_mapping == [0, -1]
                    else:
                        assert op_result_dist_attr.dims_mapping == [-1, 0]
                    assert op_value.dist_attr().partial_status == {}
                    print("split_op.value.dist_attr匹配")



       
if __name__ == '__main__':
    # TestReshardPToS().run_pir_test_case()
    TestReshardPToS().run_pir_unbalanced_split_test_case()