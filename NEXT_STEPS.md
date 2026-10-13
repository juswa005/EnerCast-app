# EnerCast — Next Steps

A checklist for rolling out the improved dashboard, forecasting model and firmware.
Work through the sections in order: **Git → Streamlit → GitHub → Arduino**.

---

## 1. Git — put the improvements on a test branch

The dashboard improvements (`app.py`, `forecasting.py`, `requirements.txt`, `.gitignore`,
`.streamlit/secrets.toml.example`) go on their own branch first, so the live site keeps
running the current version until the new one has been checked.

```bash
cd ~/Projects/ARDUINO_PROJS/EnerCast/enercast-app
git checkout -b dashboard-forecast-improvements
git add app.py forecasting.py requirements.txt .gitignore .streamlit/secrets.toml.example
git commit -m "Improve dashboard accuracy and forecasting model"
git push -u origin dashboard-forecast-improvements
```

> ⚠️ `app.py` now imports `forecasting.py`. Always commit and push them **together**,
> or the deployed app will crash on start-up.

- [ ] Branch pushed to GitHub

---

## 2. Streamlit Community Cloud — test, then go live

1. **Deploy a test copy.** On [share.streamlit.io](https://share.streamlit.io), click
   **Create app** and point it at the `dashboard-forecast-improvements` branch
   (main file: `app.py`).
2. **Check it works.** Log in with each access code (landlord and both boarders):
   - numbers look sensible (no huge kWh values)
   - charts and the hourly consumption bars load
   - in Room 1, turn on **Check model accuracy**
3. **Add secrets to the live app.** In the live app, open **⋮ → Settings → Secrets** and
   paste the settings below, filled in with your own values. Choose **new** access codes;
   the old ones are visible in the repo's public history.

   ```toml
   channel_id = "YOUR_CHANNEL_ID"
   read_key = "YOUR_NEW_THINGSPEAK_READ_KEY"
   rate_per_kwh = 11.0
   label_simulated_data = false   # true = tag rooms that show simulated data

   landlord_code = "NEW_LANDLORD_CODE"
   room1_code = "NEW_ROOM1_CODE"
   room2_code = "NEW_ROOM2_CODE"
   ```

   Any value you leave out falls back to the default in `app.py`.
4. **Go live.** Once the test copy looks good, merge into `main`. The live app redeploys
   automatically:

   ```bash
   git checkout main
   git merge dashboard-forecast-improvements
   git push
   ```
5. **Clean up.** Delete the temporary test app on Streamlit Cloud.

Notes:
- After the app has been asleep, the first load takes ~20–30 s while the models train.
  Later loads use cached results.
- `label_simulated_data = true` shows "simulated" tags on rooms with no live meter. Leave it
  `false` for presentations.

- [ ] Test app deployed and checked
- [ ] Secrets added to the live app (with new access codes)
- [ ] Merged into `main` and pushed
- [ ] Test app deleted

---

## 3. GitHub — lock the repository down

1. **Make the repo private.** Go to **Settings → General → Danger Zone → Change visibility →
   Private**. Streamlit Community Cloud can still deploy private repos; if the app loses
   access, reconnect it from your Streamlit account's GitHub settings.
2. **Rotate the ThingSpeak read key.** Making the repo private doesn't erase old commits, so
   the old key is already out there. In ThingSpeak, open your channel →
   **API Keys → Generate New Read API Key**, then put the new key **only** in Streamlit
   Secrets (step 2.3).
3. **Never commit `.streamlit/secrets.toml`.** It's already listed in `.gitignore`.

- [ ] Repo set to private
- [ ] Read key regenerated and updated in Streamlit Secrets

---

## 4. Arduino — flash and calibrate the new firmware

The firmware lives in `EnerCast_Nano/` (PlatformIO). Wi-Fi and ThingSpeak credentials are in
`EnerCast_Nano/include/secrets.h`. Keep that file out of any public repo.

1. **Upload with no appliance switched on.** At boot the device measures the sensor's own
   noise and subtracts it from every reading.
2. **Check the calibration.** Open the Serial Monitor at **9600 baud**. The
   `Noise calibration:` line should show a value **under 0.25 A**. If it says
   "too high (load on?)", switch everything off and send `C` to recalibrate.
3. **Calibrate against a clamp meter.** Run a steady load (kettle, iron, fan) and compare:

   ```cpp
   // EnerCast_Nano/src/main.cpp
   const float CURRENT_CAL = 1.00;   // set to: clamp-meter amps / EnerCast amps
   ```

   Example: the meter reads 4.00 A and EnerCast reads 4.20 A, so
   `CURRENT_CAL = 4.00 / 4.20 = 0.952`. Upload again.
4. **Reset the energy counter.** Send `R` in the Serial Monitor once, to start from 0 kWh.
   The total is now saved to EEPROM every 15 minutes and survives power cuts.
5. **Optional: tidy the channel.** In ThingSpeak, **Channel Settings → Clear Channel**
   removes the readings uploaded in the old voltage/current/power format. The dashboard
   handles them correctly already, so this only tidies things up.
6. **Let it run for a few days.** Continuous data improves the forecasts:
   - the accuracy check needs at least **90 minutes** of unbroken data
   - time-of-day patterns are only used after **2 days**

Serial Monitor commands:

| Send | Does |
|------|------|
| `R`  | Reset the energy counter to 0 kWh |
| `C`  | Recalibrate sensor noise (switch all loads off first) |

LCD status (bottom-right corner): blank = last upload OK, `*` = sending, `!` = upload failed.
The firmware reconnects to Wi-Fi after 3 failed uploads in a row.

- [ ] Firmware uploaded (with no load running)
- [ ] Noise calibration under 0.25 A
- [ ] `CURRENT_CAL` set from a clamp-meter reading
- [ ] Energy counter reset
- [ ] A few days of data collected

---

## 5. After a few days of data

- In the dashboard, turn on **Check model accuracy** and compare ARIMA, Random Forest and
  the Hybrid on **your** data. In tests on simulated data, the hybrid didn't always win;
  check this before stating it in the paper.
- Update the paper so it matches the build: Arduino **Nano**, constant **230 V** (no
  voltage sensor), and a **Streamlit** dashboard. The INA219 is a DC-only sensor and can't
  measure mains AC. If you add voltage sensing, use a ZMPT101B, or a PZEM-004T for voltage,
  current, power factor and energy in one module.

### Possible future upgrades
- Give each room its own ThingSpeak channel.
- Show boarders the forecast for their own room.
- Replace the shared access codes with real logins (e.g. `streamlit-authenticator`).
