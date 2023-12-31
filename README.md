# PhoenixOS

**PhoenixOS** is an generic framework for transparently checkpointing / restoring XPU state.

TODO: give effect of fast rasing container here

TODO: give effect of training migration here

## I. Build

### 1. Preparation docker image for specified target

TODO:

### 2. Install dependencies

```bash
apt-get install -y pkg-config
python3 -m pip install meson
python3 -m pip install ninja
```

TODO: give an architecture figure here (e.g., remoting module, POS)

## II. Example

### 1. Resnet Training

### 2. FasterTransformer Serving

### III. Development Guidance

## IV. Roadmap

1. fix meson to use g++ to compile
2. write a script to compile both pos and remoting framework automatically
3. add auto-unit test script
 