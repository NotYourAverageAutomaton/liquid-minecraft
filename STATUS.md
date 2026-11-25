# Status

Between tests. A small liquid motor network, learning with a local three-factor rule (no backprop), steers a Minecraft body. Frozen, on new 8-waypoint courses it reached 96% of waypoints against 100% for a reference controller. Three limits: its paths were 1.66 times as long; resetting its memory at each waypoint did just as well, so there is no evidence yet that its memory helps; and at a planted 60-degree control twist, online learning recovered only 8 to 21% of its untwisted rate by 60,000 ticks (at 45 degrees it recovered well, 4 of 4). It is not connected to any language model yet.

Code and simulation logs are in this repo; the real-body run captures are coming (the capture host is offline).

Next, before any terrain: compare it with plain networks of the same size and a simple adaptive controller on the same courses.
