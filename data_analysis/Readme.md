# BDD Dataset Analysis(Dockerized)

This Docker setup provides a lightweight Python 3.11 environment with all required dependencies to run Jupyter Notebooks for BDD dataset workflows.

---

## Steps to Build and Run

### Prepare the Dataset

Download the **BDD100K images and labels** from the official website:

- **Images**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- **Labels**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)

After downloading, extract it and note the path for mounting.

### Navigate to the Project Directory

```bash
cd path/to/data-analysis
```

### Build Docker Image
```bash
docker build -t data-analysis-app .
```

### Run the Docker Container (with BDD dataset mounted)

```bash
docker run -it -p 8888:8888 -v $(pwd):/app -v path/to/bdd_dataset/:/app/data data-analysis-app
```

