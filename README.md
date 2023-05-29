# AppLearner- WIP

This project targets the problem of accurately estimating resource requirements for workloads running over Red Hat OpenShift Container Platform and adjusting these estimations during the course of application execution. For a majority of real-life applications, it is notoriously difficult to manually estimate the patterns of resource consumption and thus to tune the pod CPU/memory requirements accordingly to avoid under- and overutilization. While there are attempts to solve this problem by on-the-fly monitoring and adaptively scaling pods when changes are detected, these solutions could be inefficient when the application behavior is highly dynamic. Instead, AppLearner is proactively defining the provisioning plan for an application by learning and predicting its resource consumption patterns over time.
We will use time-series CPU/Memory usage data provided by Prometheus to learn and forecast future resource usage.

## Background:
K8 is an open source software that manages, automates, deploys and hosts containerized applications.

#### What is a K8 Cluster-
A Kubernetes cluster is a set of node machines for running containerized applications. If you’re running Kubernetes, you’re running a cluster.

#### What is a K8 Node-
Nodes are comprised of physical or virtual machines on the cluster that run the pods; these “worker” machines have everything necessary to run your application containers, including the container runtime and other critical services.

#### What is a K8 Pod-
This is essentially the smallest deployable unit of the Kubernetes ecosystem. A pod specifically represents a group of one or more containers running together on your cluster.

#### What is a K8 Container-
Each container that you run is repeatable; the standardization from having dependencies included means that you get the same behavior wherever and wherever you run it.
consists of an entire runtime environment of a task. Such as the application code, needed libraries and more.

#### What is a K8 Namespace-
A virtual cluster. Namespaces allow Kubernetes to manage multiple clusters (for multiple teams or projects) within the same physical cluster.


### Prometheus:
Prometheus is an open-source monitoring system. Prometheus uses Kubernetes and collects time-series data on running pods, nodes and more. The collected metrics are CPU usage, Memory usage. The data collected gives the users an apprehensive insight into their applications whilst providing real-time metrics that would help them problem-solve.
