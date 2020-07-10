FROM rust:1.44-buster AS build

ENV RUSTFLAGS "-C target-cpu=native"

# Install HDF5
RUN apt-get update \
    && apt-get install -y libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
RUN mkdir -p /app/src/bin \
    && echo "fn main(){}" > /app/src/bin/main.rs \
    && touch /app/src/lib.rs \
    && cargo build --release \
    && cargo clean --release -p feat_extr \
    && rm -r /app/src

COPY ./src/ /app/src/
RUN cargo build --release

###################
FROM debian:buster-slim

# Install MKL
ARG MKL_VERSION=2020.1
RUN apt-get update \
    && apt-get install -y curl gnupg2 \
    && curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add - \
    && apt-get purge -y curl gnupg2 \
    && echo 'deb https://apt.repos.intel.com/mkl all main' > /etc/apt/sources.list.d/intel-mkl.list \
    && apt-get update \
    && apt-get install -y intel-mkl-64bit-${MKL_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && printf '/opt/intel/lib/intel64\n/opt/intel/mkl/lib/intel64\n' > /etc/ld.so.conf.d/intel_mkl.conf \
    && ldconfig

# Install HDF5
RUN apt-get update \
    && apt-get install -y libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/target/release/feat_extr /app

VOLUME /data

ENTRYPOINT ["/app", "--dir=/data"]
