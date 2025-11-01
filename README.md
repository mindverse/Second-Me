![Second Me](https://github.com/mindverse/Second-Me/blob/master/images/cover.png)

<div align="center">
  
[![Homepage](https://img.shields.io/badge/Second_Me-Homepage-blue?style=flat-square&logo=homebridge)](https://home.second.me/)
[![AI-native Memory](https://img.shields.io/badge/AI--native_Memory-arXiv-orange?style=flat-square&logo=academia)](https://arxiv.org/abs/2406.18312)
[![AI-native Memory 2.0](https://img.shields.io/badge/AI--native_Memory_2.0-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2503.08102)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/GpWHQNUwrg)
[![Twitter](https://img.shields.io/badge/Follow-@SecondMe_AI-1DA1F2?style=flat-square&logo=x&logoColor=white)](https://x.com/SecondMe_AI1)
[![Reddit](https://img.shields.io/badge/Join-Reddit-FF4500?style=flat-square&logo=reddit&logoColor=white)](https://www.reddit.com/r/SecondMeAI/)
[![View FAQ](https://img.shields.io/badge/FAQ-GitBook-blue?style=flat-square)](https://secondme.gitbook.io/secondme/faq)

</div>


## Our Vision

Companies like OpenAI built "Super AI" that threatens human independence. We crave individuality: AI that amplifies, not erases, **YOU**.

We’re challenging that with "**Second Me**": an open-source prototype where you craft your own **AI self**—a new AI species that preserves you, delivers your context, and defends your interests.

It’s **locally trained and hosted**—your data, your control—yet **globally connected**, scaling your intelligence across an AI network. Beyond that, it’s your AI identity interface—a bold standard linking your AI to the world, sparks collaboration among AI selves, and builds tomorrow’s truly native AI apps.

Tech enthusiasts, AI pros, domain experts, Join us! Second Me is your launchpad to extend your mind into the digital horizon.

## Key Features

### **Train Your AI Self** with AI-Native Memory ([Paper](https://arxiv.org/abs/2503.08102))
Start training your Second Me today with your own memories! Using Hierarchical Memory Modeling (HMM) and the Me-Alignment Algorithm, your AI self captures your identity, understands your context, and reflects you authentically.

 <p align="center">
  <img src="https://github.com/user-attachments/assets/a84c6135-26dc-4413-82aa-f4a373c0ff89" width="94%" />
</p>


### **Scale Your Intelligence** on the Second Me Network
Launch your AI self from your laptop onto our decentralized network—anyone or any app can connect with your permission, sharing your context as your digital identity.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a74a3f4-d8fd-41c1-8f24-534ed94c842a" width="94%" />
</p>


### Build Tomorrow’s Apps with Second Me
**Roleplay**: Your AI self switches personas to represent you in different scenarios.  
**AI Space**: Collaborate with other Second Mes to spark ideas or solve problems.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc6125c1-c84f-4ecc-b620-8932cc408094" width="94%" />
</p>

### 100% **Privacy and Control**
Unlike traditional centralized AI systems, Second Me ensures that your information and intelligence remain local and completely private.



## Getting started & staying tuned with us
Star and join us, and you will receive all release notifications from GitHub without any delay!


 <p align="center">
  <img src="https://github.com/user-attachments/assets/5c14d956-f931-4c25-b0b3-3c2c96cd7581" width="94%" />
</p>


## Quick Start

### 📊 Model Size vs. Memory (Reference Guide)

*Note: "B" in the table represents "billion parameters model". Data shown are examples only; actual supported model sizes may vary depending on system optimization, deployment environment, and other hardware/software conditions.*

| Memory (GB) | Docker Deployment (Windows/Linux) | Docker Deployment (Mac) | Integrated Setup (Windows/Linux) | Integrated Setup (Mac) |
|--------------|-----------------------------|-------------------|--------------------------|----------------|
| 8            | ~0.8B (example)                | ~0.4B (example)       | ~1.0B (example)              | ~0.6B (example)    |
| 16           | 1.5B (example)                 | 0.5B (example)        | ~2.0B (example)              | ~0.8B (example)    |
| 32           | ~2.8B (example)                | ~1.2B (example)       | ~3.5B (example)              | ~1.5B (example)    |

> **Note**: Models below 0.5B may not provide satisfactory performance for complex tasks. And we're continuously improving cross-platform support - please [submit an issue](https://github.com/mindverse/Second-Me/issues/new) for feedback or compatibility problems on different operating systems.

> **MLX Acceleration**: Mac M-series users can use [MLX](https://github.com/mindverse/Second-Me/tree/master/lpm_kernel/L2/mlx_training) to run larger models (CLI-only).

### ⚡ Get your Second Me running in just 3 steps:

#### Option 1: Automated Ubuntu/Debian Installation (Recommended for Linux)

```bash
# One-command automated installation
curl -fsSL https://raw.githubusercontent.com/mindverse/Second-Me/master/install-ubuntu.sh | bash
```

This script provides:
- ✅ Interactive installation (Docker or Local setup)
- ✅ Automatic dependency installation
- ✅ GPU detection and setup (NVIDIA CUDA)
- ✅ System optimization for your hardware
- ✅ Comprehensive error handling

📖 **[Deutsche Anleitung](INSTALLATION-DEUTSCH.md)** | **[Detailed Guide](docs/INSTALLATION-UBUNTU.md)**

#### Option 2: Manual Docker Installation (All Platforms)

```bash
# 1. Clone the repository
git clone https://github.com/mindverse/Second-Me.git
cd Second-Me
# 2. Start Docker containers
make docker-up
# 3. Access the web interface
# Open your browser and visit: http://localhost:3000
```

👉 For detailed instructions — including integrated (non-Docker) setup, model selection, memory requirements, and platform-specific tips,
check the full [Deployment Guide on GitBook](https://secondme.gitbook.io/secondme/guides/deployment).

❓ Got questions about setup, models, or any troubleshooting? [Check our FAQ](https://secondme.gitbook.io/secondme/faq).

## Tutorial and Use Cases
🛠️ Feel free to follow [User tutorial](https://secondme.gitbook.io/secondme/getting-started) to build your Second Me.

💡 Check out the links below to see how Second Me can be used in real-life scenarios:
- [Felix AMA (Roleplay app)](https://app.secondme.io/example/ama)
- [Brainstorming a 15-Day European City Itinerary (Network app)](https://app.secondme.io/example/brainstorming)
- [Icebreaking as a Speed Dating Match (Network app)](https://app.secondme.io/example/Icebreaker)


## What's Next: May 2025

Second Me continues to evolve as the open-source identity infrastructure for AI. Here's what's on deck for May:

- 🗂️ **Version Control**: Smarter versioning of memory and identity states  
- 🧠 **Continuous Training Pipelines**: Keep your AI self evolving over time, with ongoing updates based on new memory inputs.
- ⚙️ **Performance & Stability Improvements**: Enhancements across inference ability, model alignment,  and base model upgrades
- ☁️ **Cloud Solutions**: Explore cloud-based solutions for both model training (fine-tuning) and model deployment, to reduce the hardware burden on users' local machines.

## Contributing

We’d love for you to help shape what’s coming next — whether it’s fixing bugs, building new features, or improving docs.

- 📘 Check out our [Contribution Guide](./CONTRIBUTING.md) to get started  
- 💻 Submit ideas, issues, or PRs on [GitHub](https://github.com/mindverse/Second-Me)  
- 💬 Join the conversation and stay updated in our [Discord](https://discord.gg/GpWHQNUwrg) — it’s where the community lives


## Contributors

We would like to express our gratitude to all the individuals who have contributed to Second Me! If you're interested in contributing to the future of intelligence uploading, whether through code, documentation, or ideas, please feel free to submit a pull request to our repository: [Second-Me](https://github.com/Mindverse/Second-Me).


<a href="https://github.com/mindverse/Second-Me/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mindverse/Second-Me" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Acknowledgements

This work leverages the power of the open-source community. 

For data synthesis, we utilized [GraphRAG](https://github.com/microsoft/graphrag) from Microsoft.

For model deployment, we utilized [llama.cpp](https://github.com/ggml-org/llama.cpp), which provides efficient inference capabilities.

Our base models primarily come from the [Qwen2.5](https://huggingface.co/Qwen) series.

We also want to extend our sincere gratitude to all users who have experienced Second Me. We recognize that there is significant room for optimization throughout the entire pipeline, and we are fully committed to iterative improvements to ensure everyone can enjoy the best possible experience locally.

## License

Second Me is open source software licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

[license]: ./LICENSE

## Star History

<a href="https://www.star-history.com/#mindverse/Second-Me&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date" />
 </picture>
</a>
