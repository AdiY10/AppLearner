# AppLearner - WIP

AppLearner is a project that addresses the challenge of accurately estimating resource requirements for workloads running on the Red Hat OpenShift Container Platform. The project aims to adjust these estimations during the course of application execution to avoid under- and overutilization. For many real-life applications, manually estimating resource consumption patterns and tuning pod CPU and memory requirements accordingly can be difficult. Existing solutions that adaptively scale pods based on on-the-fly monitoring can be inefficient when dealing with highly dynamic application behavior. Instead, AppLearner takes a proactive approach by defining a provisioning plan for an application through learning and predicting its resource consumption patterns over time. To accomplish this, the project utilizes time-series CPU/Memory usage data provided by Prometheus to learn and forecast future resource usage.

## Background

### Kubernetes (K8)

Kubernetes is an open-source software platform used for managing, automating, deploying, and hosting containerized applications. It provides a robust framework for managing containerized workloads and services, making it easier to scale, update, and maintain applications in a distributed environment.

### Kubernetes Cluster

A Kubernetes cluster is a collection of node machines that are responsible for running containerized applications. When working with Kubernetes, you operate within the context of a cluster. The cluster manages the deployment and execution of your applications, ensuring high availability and scalability.

### Kubernetes Node

A Kubernetes node refers to a physical or virtual machine within a cluster that runs pods. Pods are the smallest deployable units of the Kubernetes ecosystem, representing one or more containers running together. Nodes provide the necessary resources and services to execute your application containers, including the container runtime and other essential components.

### Kubernetes Pod

A Kubernetes pod is the fundamental unit of execution in Kubernetes. It represents a group of one or more containers that are deployed together on a cluster. Pods enable co-location and communication between containers within the same logical application. Each pod has its own unique IP address and shares the same network namespace, allowing containers within a pod to communicate with each other using localhost.

### Kubernetes Container

A Kubernetes container is a lightweight, standalone, and executable software package that includes everything needed to run a specific task or application. Containers provide a consistent runtime environment by bundling the application code, dependencies, libraries, and configuration files together. They offer portability and reproducibility, ensuring consistent behavior regardless of the underlying infrastructure.

### Kubernetes Namespace

In Kubernetes, a namespace is a virtual cluster that provides a scope for resources. Namespaces allow you to logically divide a physical cluster into multiple virtual clusters, making it easier to manage and organize your applications. Each namespace has its own set of resources, such as pods, services, and configuration, providing isolation and separation between different teams or projects within the same physical cluster.

## Prometheus

Prometheus is an open-source monitoring system widely used in the Kubernetes ecosystem. It collects time-series data on various aspects of a Kubernetes cluster, including running pods, nodes, and more. Prometheus gathers metrics such as CPU usage and memory usage, providing real-time insights into the performance and health of applications. This data helps users troubleshoot issues, optimize resource allocation, and gain a comprehensive understanding of their applications' behavior.
