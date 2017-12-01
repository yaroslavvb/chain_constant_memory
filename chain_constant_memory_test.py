from collections import OrderedDict
from linearize import OrderedSet
import linearize as linearize_lib
import numpy as np
import os, sys, time
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import toposort

import mem_util

# workaround for https://github.com/tensorflow/tensorflow/issues/13754
setattr(tf.GraphKeys, "VARIABLES", "variables")


def recompute_tensor(target, known_values, preceding_op=None,
                     copy_known_values=False):
  """Computes target tensor from known_values. If preceding_op is not None,
  adds necessary control dependencies such that newly created computation takes
  place after preceding_op. 

  If copy_known_values is set, also copies known_values (for nicer graph
  visualization)
  """

  assert is_computable(target, known_values)
  
  # position of target in parent op
  target_pos = list(target.op.outputs).index(target)

  if copy_known_values:
    computation = ge.get_backward_walk_ops(target)
  else:
    computation = ge.get_backward_walk_ops(target, stop_at_ts=known_values)
    
  # create copy of computation
  copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(computation), {})

  # find our target tensor in the new computation
  new_target_op = info._transformed_ops[target.op]
  new_target = new_target_op.outputs[target_pos]
  new_computation = list(info._transformed_ops.values())

  # restrict computation to run after given op
  SAVE_ON_CONTROL_EDGES = True

  if SAVE_ON_CONTROL_EDGES:
    # only add "run_after" control dependencies to root of computation,
    # the rest automatically runs after because of data dependencies
    # TODO: more efficient implementation by walking back from new_target
    # instead of whole graph
    computation_graph = linearize_lib.get_graph(restrict_to=new_computation)

    # note, toposort order is reversed from networkx/mine convention
    computation_root = list(toposort.toposort(computation_graph))[-1]
    for op in computation_root:
      run_after(op, preceding_op)
  else:
    if preceding_op is not None:
      for op in info._transformed_ops.values():
        run_after(op, preceding_op)
  return new_target

def replace_input(op, old_input, new_input):
  """Replaces old input with new input in op"""
  ge.reroute_ts([new_input], [old_input], can_modify=[op])

# TODO: rename to "before", after"
def run_after(a, b):
  """Rewrites the graph to run a after b."""
  already_after = (b in a.control_inputs) or (b in [i.op for i in a.inputs])

  if already_after:
    return 0
  ge.reroute.add_control_inputs(a, [b])
  return 1


def positions(ll, item):
  """Return all positions of item in list."""
  
  start_pos = 0
  position_list = []
  try:
    while True:
      pos = ll.index(item, start_pos)
      position_list.append(pos)
      start_pos = pos+1
  except ValueError:
    pass
  return position_list


def is_computable(result, known_values):
  """Returns true if given tensor is computable from known values."""

  computable_ops = ge.get_forward_walk_ops([val.op for val in known_values])
  return result.op in computable_ops


def create_session():
  """Create session with optimizations disabled."""
  from tensorflow.core.protobuf import rewriter_config_pb2
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000,
                          graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding=rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True
  #  config.log_device_placement = True
  return tf.Session(config=config)


run_metadata = None
DO_TRACING = True
def sessrun(*args, **kwargs):
  """Helper method to use instead of sess.run that will automatically
  capture run_metadata."""
  global sess, run_metadata
  
  if not DO_TRACING:
    return sess.run(*args, **kwargs)
  
  run_metadata = tf.RunMetadata()
  kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)
  return result

def test_chain_constant_memory():
  """Test that backprop on a chain of length n takes constant memory."""
  global sess, run_metadata

  from tensorflow.python.ops import gen_math_ops
  tanh_grad = gen_math_ops._tanh_grad

  size_mbs = 1   # size of each node
  size = size_mbs * 250000  

  gg = tf.get_default_graph()
  
  tf_dev = tf.device('/cpu:0')
  tf_dev.__enter__()
  
  n = 20  
  A = [None]*(n+1)
  A[0] = tf.fill((size,), 1.0, name="A0")
  for L in range(1, n+1):
    name = "A"+str(L)
    A[L] = tf.tanh(A[L-1], name=name)

  B = [None]*(n+1)
  B[n] = tf.fill((size,), 1.0, name="B"+str(n))
    
  run_after(B[n].op, A[n].op)
  for L in range(n-1, -1, -1):
    name = "B"+str(L)
    B[L] = tanh_grad(A[L+1], B[L+1], name=name)

  # for each op, obtain steps during which any output of this op is consumed
  execution_order = linearize_lib.get_execution_order(B[0])
  consuming_schedule = OrderedDict()
  for op in gg.get_operations():
    consuming_ops = OrderedSet()  # OrderedSet for determinism
    for output in op.outputs:
      consuming_ops.update(output.consumers())
    consuming_schedule[op] = [execution_order.index(c) for c in consuming_ops]

  for step, op in enumerate(execution_order):
    for op_input in op.inputs:
      # get all the times when this input is consumed
      consume_times = consuming_schedule[op_input.op]
      assert step in consume_times

      # if it's been consumed before, save memory by recomputing it
      consumed_before = len([t for t in consume_times if t<step]) > 0
      if consumed_before:
        assert step>0
        # want recomputation to happen as late as possible, schedule to run
        # it after the op that was scheduled to execute right before this op
        prev_op = execution_order[step-1]
        new_input = recompute_tensor(op_input, known_values=[A[0]],
                                     preceding_op=prev_op)
        replace_input(op, old_input=op_input, new_input=new_input)

  sess = create_session()
  sessrun(B[0].op)
  peak_cpu = mem_util.peak_memory(run_metadata)['/cpu:0']

   # chain of length 20, backprop should use 3 MB instead of 20
   
  print("Memory to backprop on chain of length %d: %.1f MB" %(n, peak_cpu/1e6,))
  assert abs(peak_cpu - 3e6) < 1e4

def main():
  test_chain_constant_memory()
  
if __name__=='__main__':
  main()
