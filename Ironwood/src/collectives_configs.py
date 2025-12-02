import yaml
import argparse
from pathlib import Path

def main(args):
    topology = args.topology
    collective = args.collective
    output_path = args.output_path
    
    a = int(topology.split("x")[0])
    b = int(topology.split("x")[1])
    c = int(topology.split("x")[2])
    chips = a * b * c
    devices = chips * 2
    
    config = {}
    config['benchmarks'] = [{}]
    if "all_to_all" in collective:
        config['benchmarks'][0]['benchmark_name'] = "all_to_all"
    elif "all_gather" in collective:
        config['benchmarks'][0]['benchmark_name'] = "all_gather"
    elif "all_reduce" in collective:
        config['benchmarks'][0]['benchmark_name'] = "psum"
    elif "reduce_scatter" in collective:
        config['benchmarks'][0]['benchmark_name'] = "psum_scatter"
    else:
        exit(1)
        
    config['benchmarks'][0]['benchmark_sweep_params'] = []
    
    params = {}
    params['matrix_dim_range'] = {}
    params['matrix_dim_range']['start'] = 2
    params['matrix_dim_range']['multiplier'] = 2
    params['matrix_dim_range']['end'] = 8192 if chips <= 64 else 32768
    params['dtype'] = "float32"
    params['ici_size_range'] = devices
    if "_1d" in collective:
        params['op_dimension'] = 1 
    elif "_2d" in collective:
        params['op_dimension'] = 2 
    elif "_3d" in collective:
        params['op_dimension'] = 3
    else:
        exit(1)
    params['num_runs'] = 5
        
    if "all_to_all" in collective:
        if params['op_dimension'] == 1:            
            params['mesh_shape'] = str(a)+"x"+str(b)+"x"+str(c*2)
            params['sharding_strategy'] = "1x"+str(b)+"x1" # or 1xbx1?
        elif params['op_dimension'] == 2:
            params['mesh_shape'] = str(a)+"x"+str(b*c*2)
            params['sharding_strategy'] = "1x"+str(b*c*2)
        else:
            params['mesh_shape'] = str(a)+"x"+str(b)+"x"+str(c*2)
            params['sharding_strategy'] = str(a)+"x"+str(b)+"x"+str(c*2)
        config['benchmarks'][0]['benchmark_sweep_params'].append(params)
    else:
        # Parallel Replica Groups
        if params['op_dimension'] == 1:            
            params['mesh_shape'] = str(a*4)+"x"+str(b)+"x"+str(int(c/2))
            params['sharding_strategy'] = "1x"+str(b)+"x1"
        elif params['op_dimension'] == 2:
            params['mesh_shape'] = str(a)+"x"+str(b*4)+"x"+str(int(c/2))
            params['sharding_strategy'] = "1x"+str(b*c)+"x1"
        else:
            params['mesh_shape'] = str(a*b*c)+"x2"
            params['sharding_strategy'] = str(a*b*c)+"x1"         
        config['benchmarks'][0]['benchmark_sweep_params'].append(params.copy())

        # Non Parallel Replica Groups
        if params['op_dimension'] == 1:     
            params['mesh_shape'] = str(a)+"x"+str(b)+"x"+str(c*2)
            params['sharding_strategy'] = "1x"+str(b)+"x1" # or 1xbx1?
        elif params['op_dimension'] == 2:
            params['mesh_shape'] = str(a)+"x"+str(b*c*2)
            params['sharding_strategy'] = "1x"+str(b*c*2)
        else:
            params['mesh_shape'] = str(a)+"x"+str(b)+"x"+str(c*2)
            params['sharding_strategy'] = str(a)+"x"+str(b)+"x"+str(c*2)
        config['benchmarks'][0]['benchmark_sweep_params'].append(params)

    config['benchmarks'][0]['trace_dir'] = output_path+"/"+collective
    config['benchmarks'][0]['csv_path'] = output_path+"/"+collective
    config['benchmarks'][0]['xlml_metrics_dir'] = output_path+"/"+collective
    config['benchmarks'][0]['xla_dump_dir'] = output_path+"/"+collective+"/hlo_graphs"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    config_file = output_path+"/"+collective+".yaml"
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        print(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate microbenchmark collective config file."
    )
    parser.add_argument(
        "--collective",
        type=str,
        required=True,
        help="Name of the collective to generate .",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path to output.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        help="TPU supported toplogy, like 4x4x4.",
    )
    args = parser.parse_args()
    main(args)
