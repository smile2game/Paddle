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
        self._mesh = dist.ProcessMesh([0], dim_names=["x"])
        # self._out_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])

    def run_pir_test_case(self):
        paddle.enable_static()
        if self._backend == "gpu": #只测 GPU
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2 #设置参数
        SEQ_LEN = 4
        HIDDEN_SIZE = 2
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

        self.rank = dist.get_rank()
        # if self.rank == 0:
        #     ipdb.set_trace()

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
            print("测试op成功!")

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
                print("check op dist_attr!")
                #check op.dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh #只用到了一个mesh

                #check op_operand
                print("check op_operand dist_attr")
                op_operand_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr() #得到operand的分布式属性
                assert op_operand_dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.dims_mapping == [-1,-1]
                assert op_operand_dist_attr.partial_status == {0: paddle.base.core.ReduceType.kRedSum} #怎么创建一个reduce type???

                #check op_result
                print("check op_result dist_attr and value")
                op_result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr() 
                assert op_result_dist_attr.process_mesh == self._mesh
                assert op_result_dist_attr.dims_mapping == [0, -1]
                assert op_result_dist_attr.partial_status == {}
                #check op value
                op_result = op.result(0)
                # if self.rank == 0:
                #     ipdb.set_trace()
                assert op_result.is_dense_tensor_type()
                # 如何得知 op_result的所有方法
                assert op_result.is_dist_dense_tensor_type() #这里不满足,因为他是shard的
                
        


    def run_pir_unbalanced_split_test_case(self):
        paddle.enable_static()
        if self._backend == "gpu": #只测 GPU
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2 #设置参数
        SEQ_LEN = 2
        HIDDEN_SIZE = 3
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
        # if self._shard == 0:
        #     np.testing.assert_equal(main_program.num_ops(),3) #让我来猜一下数量
        #     std_ops = [
        #         "builtin.parameter",
        #         "dist_op.shard_tensor",
        #         # "pd_op.transpose", #等于0就不用转置了
        #         "pd_op.reduce_scatter",
        #         # "pd_op.transpose"
        #     ]
        #     np.testing.assert_equal(
        #         ops,
        #         std_ops,
        #     )
        #     print("测试op成功!")

        # if self._shard == 1:
        #     np.testing.assert_equal(main_program.num_ops(),5) #让我来猜一下数量
        #     std_ops = [
        #         "builtin.parameter",
        #         "dist_op.shard_tensor",
        #         "pd_op.transpose", #等于0就不用转置了
        #         "pd_op.reduce_scatter",
        #         "pd_op.transpose"
        #     ]
        #     np.testing.assert_equal(
        #         ops,
        #         std_ops,
        #     )
        
        #开始测试属性
        # for op in main_program.global_block().ops:
        #     if op.name() == "pd_op.reduce_scatter":
        #         print("check op dist_attr!")
        #         #check op.dist_attr
        #         assert op.dist_attr.num_operands() == 1
        #         assert op.dist_attr.num_results() == 1
        #         assert op.dist_attr.process_mesh == self._mesh #只用到了一个mesh

        #         #check op_operand
        #         print("check op_operand dist_attr")
        #         op_operand_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr() #得到operand的分布式属性
        #         assert op_operand_dist_attr.process_mesh == self._mesh
        #         assert op_operand_dist_attr.dims_mapping == [-1,-1]
        #         assert op_operand_dist_attr.partial_status == {0: paddle.base.core.ReduceType.kRedSum} #怎么创建一个reduce type???

        #         #check op_result
        #         print("check op_result dist_attr and value")
        #         op_result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr() 
        #         assert op_result_dist_attr.process_mesh == self._mesh
        #         assert op_result_dist_attr.dims_mapping == [0, -1]
        #         assert op_result_dist_attr.partial_status == {}
        #         #check op value
        #         op_result = op.result(0)
        #         # if self.rank == 0:
        #         #     ipdb.set_trace()
        #         assert op_result.is_dense_tensor_type()
        #         # 如何得知 op_result的所有方法
        #         assert op_result.is_dist_dense_tensor_type() #这里不满足,因为他是shard的


       
if __name__ == '__main__':
    # TestReshardPToS().run_pir_test_case()
    TestReshardPToS().run_pir_unbalanced_split_test_case()