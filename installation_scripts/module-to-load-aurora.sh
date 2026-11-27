#!/usr/bin/env bash

# ALCF-provided PyTorch (XPU), oneAPI, MPICH and mpi4py stack for Aurora.
_mattergen_restore_u=0
case "$-" in *u*) _mattergen_restore_u=1; set +u ;; esac
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

if ! command -v module >/dev/null 2>&1; then
    for _init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
        # shellcheck disable=SC1090
        [[ -f "${_init}" ]] && { source "${_init}"; break; }
    done
    unset _init
fi

module reset
module load frameworks

(( _mattergen_restore_u )) && set -u
unset _mattergen_restore_u
