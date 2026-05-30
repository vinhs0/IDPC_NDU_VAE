import os
import time
import math
import statistics

from ga.GA import GA
from ga.Configs import Configs
from problem.IDPCNDU import IDPCNDU 

def solver(fw, data_path, output_path, name):
    t1 = time.time()
    
    # Initialize task
    task = IDPCNDU()
    print(data_path)
    # Assuming readData in Java -> read_data in Python
    task.read_data(data_path)
    
    ga = GA(task, output_path, name)
    bf = float('inf')
    
    rs = []
    
    for seed in range(Configs.REPEAT):
        # Set seed on the shared random instance defined in configs.py
        Configs.rd.seed(seed)
        
        best = ga.run(seed)
        
        # In GA, fitness is stored as negative cost. Convert back to positive.
        current_cost = -best.fitness
        
        print(f"Seed {seed} Best distance: {current_cost}")
        
        bf = min(current_cost, bf)
        rs.append(current_cost)
        print("---------------------------------------")
    
    t2 = time.time()
    
    # Calculate statistics
    # Java: (t2-t1)/1000/60/Configs.REPEAT (Average minutes per run)
    total_duration_minutes = (t2 - t1) / 60
    avg_time_per_run = total_duration_minutes / Configs.REPEAT
    
    avg_val = statistics.mean(rs)
    # Handle case where REPEAT=1 to avoid div by zero in stdev
    std_val = statistics.stdev(rs) if len(rs) > 1 else 0.0
    
    # Write to result file
    # Format: Name \t BF \t AVG \t STD \t Time
    line = f"{name}\t{int(bf)}\t{avg_val:.2f}\t{std_val:.2f}\t{avg_time_per_run:.2f}\n"
    fw.write(line)

def build_model(fw, file_path, data_path):
    # Logic: Read "file.txt" inside the directory to get list of instances to run
    input_list_file = os.path.join(file_path, "file.txt")
    
    if not os.path.exists(input_list_file):
        print(f"Warning: {input_list_file} not found.")
        return

    with open(input_list_file, 'r') as sc:
        # Skip header line (sc.nextLine() in Java)
        header = sc.readline()
        
        for line in sc:
            name = line.strip()
            if not name:
                continue
            
            # Create output subdirectory for this instance
            out_dir = os.path.join(file_path, name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            # Construct data path
            dt_path = os.path.join(data_path, name + ".txt")
            
            solver(fw, dt_path, out_dir, name)

def main():
    print("Running...")
    
    # Configuration Paths
    data_path = r"D:\multi-nde-py\dataset"
    output_path = "g_output"
    result_file = os.path.join("g_output", "test.txt")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get all subdirectories in 'output'
    children = []
    if os.path.exists(output_path):
        for entry in os.scandir(output_path):
            if entry.is_dir():
                children.append(entry)

    # Sorting Logic
    def sort_key(entry):
        try:
            return entry.name.split("_")[1]
        except IndexError:
            return entry.name # Fallback

    children.sort(key=sort_key, reverse=True)
    
    # Process each folder
    for entry in children:
        # Replicating logic: FileWriter fw = new FileWriter(result, true) inside the loop
        # 'a' mode is append
        with open(result_file, "a") as fw:
            file_path = entry.path
            
            # Check if it contains ".txt" (though is_dir() check above usually handles this)
            if ".txt" in file_path:
                continue
            
            build_model(fw, file_path, data_path)
            fw.flush()
            
    print("DONE!")

if __name__ == "__main__":
    main()