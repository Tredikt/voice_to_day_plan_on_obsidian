import paramiko
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Commit and push local changes
print("Staging files...")
subprocess.run(["git", "add", "."], check=True)

commit_msg = "fix: fully support Moscow timezone for date parsing and markdown additions"
print(f"Committing with message: {commit_msg}")
subprocess.run(["git", "commit", "-m", commit_msg, "--author", "Tredikt <117392720+Tredikt@users.noreply.github.com>"], check=True)

print("Pushing to GitHub main branch...")
subprocess.run(["git", "push", "origin", "main"], check=True)
print("Pushed successfully!")

# 2. SSH deploy on the server
hostname = "109.73.202.83"
username = "root"
password = "jUb6Y9NXYP2?u+"

print("Connecting to Ubuntu server...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(hostname, username=username, password=password, timeout=30)
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

def run_command(cmd):
    print(f"\nRunning: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    
    if out:
        print("STDOUT:")
        print(out)
    if err:
        print("STDERR:")
        print(err)
    return exit_status, out, err

# Step 2.1: Update repository on server
print("Updating the repository on the server...")
run_command("cd /home/vibe_cases/voice_to_day_plan_on_obsidian && git fetch --all && git reset --hard origin/main")

# Step 2.2: Rebuild/restart the container to apply changes
print("Rebuilding and starting docker containers...")
status, out, err = run_command("cd /home/vibe_cases/voice_to_day_plan_on_obsidian && docker compose down && docker compose up -d --build")
if status != 0:
    run_command("cd /home/vibe_cases/voice_to_day_plan_on_obsidian && docker-compose down && docker-compose up -d --build")

# Step 2.3: Check logs
run_command("cd /home/vibe_cases/voice_to_day_plan_on_obsidian && (docker compose ps || docker-compose ps)")
run_command("cd /home/vibe_cases/voice_to_day_plan_on_obsidian && (docker compose logs --tail=15 || docker-compose logs --tail=15)")

ssh.close()
print("\nDeployment successfully finished!")
