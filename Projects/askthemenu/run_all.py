import os
os.system("python src/generate_data.py")
os.system("python src/embed.py")
os.system("python src/cluster.py")
os.system("python src/build_index.py")
