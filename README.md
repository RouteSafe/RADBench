# RADBench

This repository provides a comprehensive evaluation framework for various BGP (Border Gateway Protocol) anomaly detection methods. The goal of this project is to assess the performance of different detection algorithms and provide a benchmark for future research in this area.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Contact](#contact)

## Introduction

Border Gateway Protocol (BGP) is the routing protocol that makes the Internet work across different networks. However, BGP is susceptible to various anomalies such as route hijacking, leak, and outage. Accurate detection of these anomalies is crucial for maintaining the stability and security of the Internet. This repository contains implementations of state-of-the-art BGP anomaly detection methods and provides a framework for evaluating their effectiveness.

## Dataset

We have uploaded data for 38 anomaly routing events to https://pan.cstcloud.cn/web/share.html?hash=I6DjC3rRs8. The data includes routing information for six hours before and after each anomaly event. Each event's data is contained in a separate folder, and specific information about each event can be found in the file named anomaly-event-info.csv.


## Methods

The repository includes implementations of the following BGP anomaly detection methods:

1. **Rule-Baed Models**: ARTEMIS, CAIDA AS Relationship
2. **ML-Based Models**: RoLL+, Ada.Boost, MLP and SVM 
3. **DL-Based Models**: ISP Self-Operated, BEAM, RNN and MSLSTM

You can find the relevant code for Ada.Boost, MLP, SVM, RNN, and MSLSTM in the MSLSTM . Each method's file includes the environment required to run the method, as well as the code for detection and evaluation.

## Contact

If you have any questions, you can contact xiaolan4279@proton.me