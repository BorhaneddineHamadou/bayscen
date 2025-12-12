#!/bin/bash

set PYTHONPATH=%PYTHONPATH%;leaderboard
set PYTHONPATH=%PYTHONPATH%;leaderboard\team_code

set LEADERBOARD_ROOT=leaderboard
set CHALLENGE_TRACK_CODENAME=SENSORS
set PORT=2000 # same as the carla server port
set TM_PORT=2500 # port for traffic manager, required when spawning multiple servers/clients
set DEBUG_CHALLENGE=0
set REPETITIONS=1 # multiple evaluation runs
set ROUTES=leaderboard\data\training_routes\routes_town03_long.xml
set TEAM_AGENT=leaderboard\team_code\interfuser_agent.py # agent
set TEAM_CONFIG=leaderboard\team_code\interfuser_config.py # model checkpoint, not required for expert
set CHECKPOINT_ENDPOINT=results\sample_result.json # results file
set SCENARIOS=leaderboard\data\scenarios\town03_all_scenarios.json
set SAVE_PATH=data\eval # path for saving episodes while evaluating
set RESUME=True

python ${LEADERBOARD_ROOT}\leaderboard\leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

