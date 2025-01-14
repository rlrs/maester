import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO, 
                   format='%(message)s', handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self, job_dir: str | Path):
        self.job_dir = Path(job_dir)
        if not (self.job_dir / "slurm.sh").exists():
            raise ValueError(f"No slurm.sh found in {job_dir}")
        
        # Initialize SQLite database
        self.db_path = self.job_dir / "job_manager.db"
        self.init_db()
        
        # Configuration
        self.submit_interval = 60 * 60 # 1 hour
        self.check_interval = 30
        self.cancel_threshold = 60 * 10
        self.status_interval = 30
        
        # Scan existing state
        self.sync_existing_state()
        
    def init_db(self):
        """Initialize SQLite database for job tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                submit_time TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                exit_code INTEGER,
                processed BOOLEAN DEFAULT FALSE  -- Track if we've processed this log/job
            )
            """)
    
    def get_queue_info(self) -> Dict[str, Dict]:
        """Get information about all jobs in the SLURM queue"""
        try:
            result = subprocess.run(
                ["squeue", "-o", "%i %j %t %S %M %L %Q", "--noheader"], 
                capture_output=True, text=True, check=True
            )
            
            jobs = {}
            for line in result.stdout.splitlines():
                job_id, name, state, start_time, time_used, time_left, priority = line.split()
                jobs[job_id] = {
                    'name': name,
                    'state': state,
                    'start_time': start_time,
                    'time_used': time_used,
                    'time_left': time_left,
                    'priority': int(priority)
                }
            return jobs
        except subprocess.CalledProcessError:
            logger.error("Failed to get queue info")
            return {}

    def get_our_jobs(self) -> List[str]:
        """Get list of our job IDs from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT job_id FROM jobs WHERE end_time IS NULL")
            return [row[0] for row in cursor.fetchall()]

    def is_job_healthy(self, job_id: str) -> bool:
        """Check if a running job is healthy by examining its log files"""
        log_file = self.job_dir / "logs" / f"{job_id}.out"
        if not log_file.exists():
            return False
        
        # Check if log has been updated recently
        if time.time() - log_file.stat().st_mtime > 600:  # No updates in 10 minutes
            return False
        
        return True
    
    @staticmethod
    def parse_slurm_time(time_str: str) -> timedelta:
        """convert slurm time format to timedelta"""
        if '-' in time_str:
            days, rest = time_str.split('-')
            days = int(days)
        else:
            days = 0
            rest = time_str
        
        parts = rest.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return timedelta(days=days, hours=h, minutes=m, seconds=s)
        elif len(parts) == 2:
            m, s = map(int, parts)
            return timedelta(days=days, minutes=m, seconds=s)
        raise ValueError(f"Invalid time string: {time_str}")

    def find_running_jobs(self) -> List[Tuple[str, timedelta]]: # (job_id, runtime)
        """Find any currently running jobs if any"""
        queue_info = self.get_queue_info()
        our_jobs = self.get_our_jobs()
        
        running = []
        for job_id in our_jobs:
            if job_id in queue_info and queue_info[job_id]['state'] == 'R':
                if self.is_job_healthy(job_id):
                    runtime = self.parse_slurm_time(queue_info[job_id]['time_used'])
                    running.append((job_id, runtime))
        return running

    def should_cancel_job(self, job_id: str, queue_info: Dict) -> bool:
        """Determine if a queued job should be cancelled"""
        if job_id not in queue_info:
            return False
            
        running_jobs = self.find_running_jobs()
        if not running_jobs:
            return False  # Keep all queued jobs if nothing is running
            
        job_info = queue_info[job_id]

        # if running, only stay alive if we're the oldest healthy job 
        if job_info['state'] == 'R':
            if len(running_jobs) > 1:
                running_jobs.sort(key=lambda x: x[1], reverse=True)
                oldest_job_id = running_jobs[0][0]
                if job_id != oldest_job_id:
                    return True
            return False
        
        # otherwise, if we're pending and about to start, cancel if there's a healthy runner
        if (job_info['state'] == 'PD' and 
            job_info['start_time'] != 'N/A' and
            running_jobs):  # any healthy running job means cancel
            
            try:
                start_time = datetime.strptime(job_info['start_time'], '%Y-%m-%dT%H:%M:%S')
                if (start_time - datetime.now()).total_seconds() < self.cancel_threshold:
                    return True
            except ValueError:
                pass
                
        return False

    def sync_existing_state(self):
        """Scan for existing logs and running jobs on startup"""
        # Get all job IDs from log files
        log_jobs = {p.stem for p in (self.job_dir / "logs").glob("*.out")}
        
        # Get current queue state
        queue_info = self.get_queue_info()
        job_name = self.job_dir.name
        
        # Find all queued jobs that belong to us
        our_queued_jobs = {
            job_id: info for job_id, info in queue_info.items()
            if info['name'] == job_name and info['state'] == 'PD'
        }
        
        logger.info(f"Found {len(our_queued_jobs)} existing queued jobs for {job_name}")
        
        # Add queued jobs to our database
        with sqlite3.connect(self.db_path) as conn:
            for job_id, info in our_queued_jobs.items():
                exists = conn.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                
                if not exists:
                    conn.execute("""
                    INSERT INTO jobs (job_id, status, submit_time)
                    VALUES (?, 'QUEUED', datetime('now'))
                    """, (job_id,))
        
        # Check each log file's job
        for job_id in log_jobs:
            # If job isn't in our DB yet, add it
            with sqlite3.connect(self.db_path) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                
                if not exists:
                    # Job in queue -> running or pending
                    if job_id in queue_info:
                        state = queue_info[job_id]['state']
                        conn.execute("""
                        INSERT INTO jobs (job_id, status, submit_time, start_time)
                        VALUES (?, ?, datetime('now'), ?)
                        """, (job_id, state, 
                             datetime.now().isoformat() if state == 'R' else None))
                    else:
                        # Job not in queue -> completed or failed
                        # Try to get exit status from sacct
                        sacct = subprocess.run(
                            ["sacct", "-j", job_id, "-o", "State,ExitCode", "--noheader"],
                            capture_output=True, text=True
                        )
                        if sacct.returncode == 0 and sacct.stdout.strip():
                            state, exit_code = sacct.stdout.split()[:2]
                            conn.execute("""
                            INSERT INTO jobs 
                            (job_id, status, submit_time, start_time, end_time, exit_code, processed)
                            VALUES (?, ?, datetime('now'), datetime('now'), datetime('now'), ?, TRUE)
                            """, (job_id, state, exit_code))

    def submit_job(self) -> Optional[str]:
        """Submit a new job using the slurm.sh script"""
        try: # TODO: backoff if jobs are failing?
            # Submit with job name matching directory
            result = subprocess.run(
                ["sbatch", f"--job-name={self.job_dir.name}", str(self.job_dir / "slurm.sh")],
                capture_output=True, text=True, check=True
            )
            
            job_id = result.stdout.strip().split()[-1]
            
            # Record in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                INSERT INTO jobs (job_id, submit_time, status)
                VALUES (?, datetime('now'), 'QUEUED')
                """, (job_id,))
                
            logger.info(f"Submitted job {job_id}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Job submission failed: {e.stderr}")
            return None

    def cancel_job(self, job_id: str):
        """Cancel a job and update database"""
        try:
            subprocess.run(["scancel", job_id], check=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                UPDATE jobs 
                SET status = 'CANCELLED', end_time = datetime('now')
                WHERE job_id = ?
                """, (job_id,))
                
            logger.info(f"Cancelled job {job_id}")
            
        except subprocess.CalledProcessError:
            logger.error(f"Failed to cancel job {job_id}")

    def update_job_states(self):
        """Update database with current state of all our jobs"""
        queue_info = self.get_queue_info()
        our_jobs = self.get_our_jobs()
        
        with sqlite3.connect(self.db_path) as conn:
            for job_id in our_jobs:
                if job_id in queue_info:
                    state = queue_info[job_id]['state']
                    if state == 'R' and conn.execute(
                        "SELECT start_time FROM jobs WHERE job_id = ?", 
                        (job_id,)).fetchone()[0] is None:
                        # Job just started running
                        conn.execute("""
                        UPDATE jobs 
                        SET status = 'RUNNING', start_time = datetime('now')
                        WHERE job_id = ?
                        """, (job_id,))
                else:
                    # Job not in queue - check if it completed successfully
                    sacct = subprocess.run(
                        ["sacct", "-j", job_id, "-o", "State,ExitCode", "--noheader"],
                        capture_output=True, text=True
                    )
                    
                    if sacct.returncode == 0 and sacct.stdout.strip():
                        state, exit_code = sacct.stdout.split()[:2]
                        conn.execute("""
                        UPDATE jobs 
                        SET status = ?, end_time = datetime('now'), exit_code = ?
                        WHERE job_id = ? AND end_time IS NULL
                        """, (state, exit_code, job_id))

    def get_queue_stats(self) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
        """Get running and queued jobs with their info"""
        queue_info = self.get_queue_info()
        our_jobs = self.get_our_jobs()
        
        running_jobs = [
            (job_id, info) for job_id, info in queue_info.items()
            if job_id in our_jobs and info['state'] == 'R'
        ]
        
        queued_jobs = [
            (job_id, info) for job_id, info in queue_info.items()
            if job_id in our_jobs and info['state'] == 'PD'
        ]
        
        # Sort queued jobs by priority
        queued_jobs.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        return running_jobs, queued_jobs

    def get_recent_completions(self, limit: int = 3) -> List[Tuple[str, str, int]]:
        """Get recent job completions/failures"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("""
                SELECT job_id, status, exit_code 
                FROM jobs 
                WHERE end_time IS NOT NULL 
                ORDER BY end_time DESC LIMIT ?
            """, (limit,)).fetchall()

    def log_status(self):
        """Log current state of running and queued jobs"""
        running_jobs, queued_jobs = self.get_queue_stats()
        
        if running_jobs:
            table = Table(title="Running Jobs")
            table.add_column("Job ID")
            table.add_column("Runtime") 
            table.add_column("Health")
            
            for job_id, info in running_jobs:
                health = "HEALTHY" if self.is_job_healthy(job_id) else "UNHEALTHY"
                table.add_row(
                    job_id,
                    info['time_used'],
                    f"[green]{health}" if health == "HEALTHY" else f"[red]{health}"
                )
            console.print(table)
            
        if queued_jobs:
            table = Table(title=f"Queued Jobs ({len(queued_jobs)})")
            table.add_column("Job ID")
            table.add_column("Priority")
            table.add_column("Start Time")
            
            for job_id, info in queued_jobs[:5]:
                table.add_row(job_id, str(info['priority']), info['start_time'])
            console.print(table)
        
        # Log recent completions
        recent = self.get_recent_completions()
        if recent:
            table = Table(title="Recent Completions")
            table.add_column("Job ID")
            table.add_column("Status")
            table.add_column("Exit Code")
            for job_id, status, exit_code in recent:
                table.add_row(job_id, status, exit_code)
            console.print(table)

    def main_loop(self):
        """Main monitoring and management loop"""
        last_submit = 0
        last_status = 0
        
        logger.info(f"Starting job manager for {self.job_dir.name}")
        self.log_status()  # Initial status
        
        while True:
            try:
                # Update job states
                self.update_job_states()
                
                # Check queue and cancel jobs if needed
                queue_info = self.get_queue_info()
                our_jobs = self.get_our_jobs()
                
                for job_id in our_jobs:
                    if self.should_cancel_job(job_id, queue_info):
                        self.cancel_job(job_id)
                        logger.info(f"Cancelled job {job_id} as it was about to start and we have a healthy running job")
                
                # Submit new jobs if needed
                if time.time() - last_submit > self.submit_interval:
                    queued_count = sum(1 for j in our_jobs 
                                     if j in queue_info and queue_info[j]['state'] == 'PD')
                    target_queued = max(1, int(172800 / self.submit_interval))
                    
                    if queued_count < target_queued:
                        logger.info(f"Submitting new job (have {queued_count}, want {target_queued})")
                        self.submit_job()
                    last_submit = time.time()
                
                # Periodic status logging
                if time.time() - last_status > self.status_interval:
                    self.log_status()
                    last_status = time.time()
                
                # Sleep
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=1)
                time.sleep(self.check_interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SLURM Job Manager")
    parser.add_argument("job_dir", help="Directory containing slurm.sh and logs/")
    args = parser.parse_args()
    
    manager = JobManager(args.job_dir)
    manager.main_loop()