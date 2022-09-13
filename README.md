# AppLearner

The goal of the project is to build as accurately as possible a prediction model, which will be able to predict the resource consumption of a given container in K8.
Time-series data on CPU/Memory usage provided by Prometheus, will be used to learn and forecast future resources usage.

## Background:
K8 is an open source software that manages, automates, deploys and hosts containerized applications.

#### What is a K8 Cluster
A Kubernetes cluster is a set of node machines for running containerized applications. If you’re running Kubernetes, you’re running a cluster.

#### What is a K8 Node
Nodes are comprised of physical or virtual machines on the cluster that run the pods; these “worker” machines have everything necessary to run your application containers, including the container runtime and other critical services.

#### What is a K8 Pod
This is essentially the smallest deployable unit of the Kubernetes ecosystem. A pod specifically represents a group of one or more containers running together on your cluster.

#### What is a K8 Container
Each container that you run is repeatable; the standardization from having dependencies included means that you get the same behavior wherever and wherever you run it.
consists of an entire runtime environment of a task. Such as the application code, needed libraries and more.

#### What is a K8 Namespace
A virtual cluster. Namespaces allow Kubernetes to manage multiple clusters (for multiple teams or projects) within the same physical cluster.


### Prometheus:
Prometheus is an open-source monitoring system. Prometheus uses Kubernetes and collects time-series data on running pods, nodes and more. The collected metrics are CPU usage, Memory usage. The data collected gives the users an apprehensive insight into their applications whilst providing real-time metrics that would help them problem-solve.
### Installation
```
$ python3 install matplotlib
$ python3 install pandas
$ python3 install sktime
$ python3 install numpy
$ python3 install torch
```
