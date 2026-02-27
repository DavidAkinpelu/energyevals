#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  gcc g++ gfortran git patch wget pkg-config \
  liblapack-dev libmetis-dev \
  glpk-utils libglpk-dev build-essential \
  unzip libblas-dev cmake \
  coinor-cbc coinor-libcbc-dev

apt-get clean
rm -rf /var/lib/apt/lists/*

workdir=/tmp

build_third_party() {
  local repo_name="$1"
  local get_script="$2"

  cd "$workdir"
  rm -rf "$repo_name"
  git clone "https://github.com/coin-or-tools/${repo_name}.git"
  cd "$repo_name"
  "./${get_script}"
  ./configure
  make -j"$(nproc)"
  make install
}

build_third_party "ThirdParty-ASL" "get.ASL"
build_third_party "ThirdParty-Mumps" "get.Mumps"

cd "$workdir"
rm -rf Ipopt

git clone --depth=1 --branch releases/3.14.17 https://github.com/coin-or/Ipopt.git
cd Ipopt
mkdir -p build
cd build

../configure --prefix=/usr/local
make -j"$(nproc)"
make install

# Update library paths for the installed libraries
printf '/usr/local/lib\n' > /etc/ld.so.conf.d/ipopt.conf
ldconfig

# Verify installation
ipopt_path="$(command -v ipopt || true)"
if [[ -z "${ipopt_path}" ]]; then
  echo "Ipopt executable not found on PATH after install." >&2
  exit 1
fi

if ! pkg-config --modversion ipopt >/dev/null 2>&1; then
  echo "pkg-config did not report an Ipopt version." >&2
  exit 1
fi

# Configure Pyomo to use the installed solver for the invoking user.
pyomo_user="${SUDO_USER:-root}"
pyomo_home="$(getent passwd "${pyomo_user}" | cut -d: -f6)"
if [[ -z "${pyomo_home}" ]]; then
  pyomo_home="/root"
fi

mkdir -p "${pyomo_home}/.python/pyomo"
cat <<EOF > "${pyomo_home}/.python/pyomo/config.json"
[solvers]
glpk=/usr/bin/glpsol
glpk_options=""
ipopt=${ipopt_path}
ipopt_options=""
EOF

# Clean up
cd /
rm -rf /tmp/Ipopt /tmp/ThirdParty-ASL /tmp/ThirdParty-Mumps
