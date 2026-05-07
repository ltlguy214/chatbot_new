# PowerShell commands to re-run failing test cases
& .\.venv312\Scripts\python.exe chatbot\test_e2e.py test_failure_runs\failing_TC_139.csv test_failure_runs\report_TC_139.csv
& .\.venv312\Scripts\python.exe chatbot\test_e2e.py test_failure_runs\failing_TC_140.csv test_failure_runs\report_TC_140.csv
