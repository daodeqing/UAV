param(
    [string]$Dataset = "UNSW",
    [string]$Device = "cuda",
    [string]$DeviceId = "0",
    [int]$NumClients = 50,
    [int]$Rounds = 100,
    [int]$Times = 3,
    [float]$JoinRatio = 1.0,
    [float]$ClientActivityRate = 1.0,
    [int]$EvalGap = 5,
    [int]$Seed = 10
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$mainPy = Join-Path $scriptRoot "main.py"

$baseArgs = @(
    $mainPy,
    "-algo", "HGFIDSUS",
    "-data", $Dataset,
    "-dev", $Device,
    "-did", $DeviceId,
    "-nc", $NumClients,
    "-jr", $JoinRatio,
    "-car", $ClientActivityRate,
    "-gr", $Rounds,
    "-eg", $EvalGap,
    "-t", $Times,
    "--seed", $Seed,
    "-lr", "0.01",
    "--dynamic_pricing", "True",
    "--warmup_rounds", "8",
    "--auction_winners_frac", "0.25",
    "--delta_server_lr", "0.05",
    "--global_base_alpha", "0.05",
    "--peer_mixing", "0.30",
    "--label_smoothing", "0.01",
    "--head_train_epochs", "2",
    "--delta_clip_norm", "1.0",
    "--enable_contracts", "True",
    "--contract_start_round", "8",
    "--min_full_clients", "4",
    "--enable_sticky_sampling", "True",
    "--sticky_fraction", "0.50",
    "--no_replace_window", "3",
    "--enable_delta_compression", "True",
    "--delta_topk", "0.10",
    "--delta_error_feedback", "True",
    "--proto_fp16", "True",
    "--staleness_decay_gamma", "0.10"
)

$experiments = @(
    @{
        Goal = "scar_full"
        Extra = @()
    },
    @{
        Goal = "ablate_no_contract"
        Extra = @("--enable_contracts", "False")
    },
    @{
        Goal = "ablate_no_sampling"
        Extra = @("--enable_sticky_sampling", "False", "--sticky_fraction", "0.0", "--no_replace_window", "0")
    },
    @{
        Goal = "ablate_no_compress"
        Extra = @("--enable_delta_compression", "False", "--delta_topk", "1.0", "--delta_error_feedback", "False", "--proto_fp16", "False")
    },
    @{
        Goal = "ablate_no_failover"
        Extra = @("--use_failover", "False")
    },
    @{
        Goal = "ablate_all_off"
        Extra = @(
            "--enable_contracts", "False",
            "--enable_sticky_sampling", "False",
            "--sticky_fraction", "0.0",
            "--no_replace_window", "0",
            "--enable_delta_compression", "False",
            "--delta_topk", "1.0",
            "--delta_error_feedback", "False",
            "--proto_fp16", "False",
            "--use_failover", "False"
        )
    }
)

foreach ($exp in $experiments) {
    Write-Host ""
    Write-Host "===== Running $($exp.Goal) ====="
    $cmd = @($baseArgs) + @("-go", $exp.Goal) + $exp.Extra
    & python @cmd
}
