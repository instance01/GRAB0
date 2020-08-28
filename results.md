Bad = B; Ok = K; Good = G

| Parameters   | TF Event file | Minutes | Result | Comments |
| ------------ | ------------- | ------- | ------ | -------- |
| BT10 | | | No | BT = Bandittest. No means that it was not successful. |
| BT11 | | | No | |
| BT12 | | | No | |
| BT13 | | | Yes | This also had full on greedy training after episode 200. Also `grad_bandit_init_random=false`. A second run had limited success (a few bad evals). |
| BT14 | | | No | |
| BT15 | | | Yes | Same comments as BT13. A few bad evals in between. |
| BT16 | | | Yes | Same comments as BT13. A second run had more variance and limited success (a few bad evals). |
| BT17 | | | No | |
| BT18 | | | No | |
| BT19 | | | No | A few good evals. |
| BT20 | | | Yes | Excellent solutions and no bad evals. 4 runs, all perfect. |
| BT21 | | | Yes | Excellent solutions, but quite often bad evals. In all 4 runs. |
| 158 | `Aug12-12:36:52-687-goshenit.cip.ifi.lmu.de-16x16-158` | | 8/10 | From here on `game=16x16`. Want: 50 episodes of good evaluations. E.g. here: 8 times that criterion was fulfilled. |
| 159 | `Aug12-12:26:14-453-heliodor.cip.ifi.lmu.de-16x16-159` | | 7/8 | |
| 160 | `Aug12-12:23:39-808-danburit.cip.ifi.lmu.de-16x16-160` | | 9/10 | |
| 161 | `Aug12-16:13:19-139-petalit.cip.ifi.lmu.de-16x16-161` | | 7/10 | 8/10 on a second run. |
| 162 | `Aug12-21:26:29-443-lapislazuli.cip.ifi.lmu.de-16x16-162` | | 10/10 | Close one. |
| 163 | `Aug12-21:26:29-302-zirkon.cip.ifi.lmu.de-16x16-163` | | 7/10 | |
| 164 | `Aug12-21:26:29-440-saphir.cip.ifi.lmu.de-16x16-164` | | 6/10 | |
| 165 | `Aug12-21:30:26-914-leucit.cip.ifi.lmu.de-16x16-165` | | 10/10 | Comment for all the above: They all diverge after a while (but I didn't use reduce on good eval scheduler this time, which most likely would've fixed the issue). Scheduler used: Exponential. |
| 166 | `` | | | From here on `game=mtcar`. Had to restart all (but left out 168, 175) because didn't go anywhere. Restarted with registry system. Below up to 188 do not include tau and init with log. |
| 167 | `` | | | |
| 168 | `Aug12-21:32:58-921-peridot.cip.ifi.lmu.de-mtcar-168` | | 0/2 | See comment at:166. |
| 169 | `` | | | |
| 170 | `` | | | One good run. Keep this one in mind. |
| 171 | `` | | | |
| 172 | `` | | | |
| 173 | `` | | | |
| 174 | `` | | | |
| 175 | `Aug13-18:43:28-872-thulit.cip.ifi.lmu.de-mtcar-175` | | 0/2 | See comment at 166. |
| 176 | `Aug16-06:14:35-136-rhodonit.cip.ifi.lmu.de-mtcar-176` | | 0/1 | Infeasable, takes too long. |
| 177 | `Aug16-06:14:35-733-smaragd.cip.ifi.lmu.de-mtcar-177` | | 0/1 | See 176 |
| 178 | `Aug16-06:14:35-921-saphir.cip.ifi.lmu.de-mtcar-178` | | | See 176 |
| 179 | `` | | | |
| 180 | `` | | | |
| 181 | `` | | | |
| 182 | `` | | | This and next 3: memory capacity increased. |
| 183 | `` | | | |
| 184 | `` | | | `simulations=1000` |
| 185 | `` | | | |
| 186 | `` | | | |
| 187 | `` | | | |
| 188 | `` | | | |
| 189 | `` | | | |
| cart1-cart47 | `runs_grab3_cart.tar.xz`| | | cart38 is the favorite, learns 20/20. Others include cart40, cart41. |
| 190 | `` | | | |
| 191 | `` | | | |
