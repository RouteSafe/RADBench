# RADBench

This repository provides a comprehensive evaluation framework for various BGP (Border Gateway Protocol) anomaly detection methods. The goal of this project is to assess the performance of different detection algorithms and provide a benchmark for future research in this area.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Setup](#setup)
- [Running the Evaluation](#running-the-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Border Gateway Protocol (BGP) is the routing protocol that makes the Internet work across different networks. However, BGP is susceptible to various anomalies such as route hijacking, leaks, and outages. Accurate detection of these anomalies is crucial for maintaining the stability and security of the Internet. This repository contains implementations of state-of-the-art BGP anomaly detection methods and provides a framework for evaluating their effectiveness.

## Dataset

The evaluation is conducted using a dataset that includes a diverse set of BGP anomaly events. 

### Dataset Features

- **Volume**: The dataset contains over 4 billion routing messages.
- **Diversity**: It includes 3 types of BGP anomalies, including hijacking, leak and outage, spanning two decades.
- **Labels**: Each event has associated whith its event info, containing the type of abnormal event, start time, end time, victim AS, attacker AS, affected prefix, and original report/paper link..

## Methods

The repository includes implementations of the following BGP anomaly detection methods:

1. **Rule-Baed Models**: ARTEMIS, CAIDA AS Relationship
2. **ML-Based Models**: RoLL+, Ada.Boost, MLP and SVM 
3. **DL-Based Models**: ISP Self-Operated, BEAM, RNN and MSLSTM

## Setup

To set up the evaluation framework, follow these steps:

1. Clone the repository:
2. Dectection
3. Running the Evaluation
