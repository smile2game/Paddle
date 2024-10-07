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

# 全局列表，用于存储所有已注册的重新切分函数
_g_reshard_func_list = []

# 定义一个用于重新切分的基础类
class ReshardFunction:
    # 检查重新切分函数是否适用于给定的张量和属性的方法
    def is_suitable(self, dist_tensor, dist_attr):
        raise NotImplementedError

    # 从源到目标执行重新切分操作的方法
    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        raise NotImplementedError

# 根据属性从列表中选择合适的重新切分函数
# 返回第一个合适的函数
def choose_reshard_func(src_dist_attr, dst_dist_attr):
    global _g_reshard_func_list
    for reshard_func in _g_reshard_func_list:
        if reshard_func.is_suitable(src_dist_attr, dst_dist_attr):
            return reshard_func
    return None

# 将新的重新切分函数注册到全局列表中的函数
# 允许动态扩展功能
def register_reshard_func(reshard_func):
    global _g_reshard_func_list
    _g_reshard_func_list.append(reshard_func)

# 清除全局列表中所有已注册的重新切分函数的函数
# 用于重置环境
def clean_reshard_funcs():
    global _g_reshard_func_list
    _g_reshard_func_list.clear()

# 检查分布属性是否表示一个分片张量的函数
# 如果任一维度不是复制的（即不等于 -1），则返回 True
def is_shard(dist_attr):
    for v in dist_attr.dims_mapping:
        if v != -1:
            return True
    return False

# 检查分布属性是否表示一个部分复制张量的函数
# 如果张量具有任何部分状态，则返回 True
def is_partial(dist_attr):
    if len(dist_attr.partial_status) > 0:
        return True
    return False

# 检查分布属性是否表示一个完全复制的张量的函数
# 如果没有部分状态且所有维度均为 -1 或为空，则返回 True
def is_replicated(dist_attr):
    dims_mapping_set = set(dist_attr.dims_mapping)
    if len(dist_attr.partial_status) == 0 and (
        len(dims_mapping_set) == 0
        or (len(dims_mapping_set) == 1 and -1 in dims_mapping_set)
    ):
        return True
    return False

# 创建张量分布属性的副本，并可以选择性地进行修改的函数
# 允许在保持其他属性不变的情况下修改进程网格、维度映射和部分状态

def copy_dist_attr_with_new_member(
    src_dist_attr,
    new_process_mesh=None,
    new_dims_mapping=None,
    new_partial_status=None,
):
    if new_process_mesh is None:
        new_process_mesh = src_dist_attr.process_mesh
    if new_dims_mapping is None:
        new_dims_mapping = src_dist_attr.dims_mapping
    if new_partial_status is None:
        new_partial_status = src_dist_attr.partial_status

    return paddle.base.libpaddle.pir.create_tensor_dist_attribute(
        new_process_mesh,
        new_dims_mapping,
        new_partial_status,
    )

# 创建操作属性的副本并可以选择性地进行修改的函数
# 允许在修改某些成员的情况下创建新的操作分布属性

def copy_op_attr_with_new_member(
    src_dist_attr,
    new_process_mesh=None,
    new_operands=None,
    new_results=None,
    new_chunk_id=None,
):
    if new_process_mesh is None:
        new_process_mesh = src_dist_attr.process_mesh
    if new_operands is None:
        new_operands = src_dist_attr.operands()
    if new_results is None:
        new_results = src_dist_attr.results()
    if new_chunk_id is None:
        new_chunk_id = src_dist_attr.chunk_id

    return paddle.base.libpaddle.pir.create_op_dist_attribute(
        new_process_mesh,
        new_operands,
        new_results,
        new_chunk_id,
    )

# 创建进程网格副本并可以选择性地进行修改的函数
# 允许修改进程网格的形状、进程 ID 和维度名称

def copy_process_mesh_with_new_member(
    src_process_mesh,
    new_shape=None,
    new_process_ids=None,
    new_dim_names=None,
):
    if new_shape is None:
        new_shape = src_process_mesh.shape
    if new_process_ids is None:
        new_process_ids = src_process_mesh.process_ids
    if new_dim_names is None:
        new_dim_names = src_process_mesh.dim_names

    return paddle.base.libpaddle.pir.create_process_mesh(
        new_shape,
        new_process_ids,
        new_dim_names,
    )
