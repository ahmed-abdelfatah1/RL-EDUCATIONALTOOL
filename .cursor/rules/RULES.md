# RL Educational Tool - Project Rules

## 🎯 PROJECT OVERVIEW
Build a **web-based interactive RL learning tool** for exploring 7 algorithms across 7 environments with real-time visualization.

## 🏗️ ARCHITECTURE (MANDATORY)

rl-edu-tool/
├── backend/ # FastAPI + Gymnasium
└── frontend/ # React + TypeScript + MUI (single page)

text

**Backend venv:** `backend/venv/`
**Frontend:** `frontend/node_modules/`

## 🔧 BACKEND STRUCTURE (EXACT)

backend/app/
├── core/config.py
├── api/v1/routes/
│ ├── environments.py
│ ├── algorithms.py
│ └── training.py
├── domain/
│ ├── environments/
│ │ ├── base_env.py # ALL envs inherit
│ │ ├── gridworld_env.py
│ │ ├── cartpole_env.py
│ │ ├── mountaincar_env.py
│ │ ├── frozenlake_env.py
│ │ ├── breakout_env.py
│ │ └── gym4real_env.py
│ ├── algorithms/
│ │ ├── base_agent.py
│ │ ├── policy_iteration.py
│ │ ├── value_iteration.py
│ │ ├── monte_carlo.py
│ │ ├── td.py
│ │ ├── n_step_td.py
│ │ ├── sarsa.py
│ │ └── q_learning.py
│ └── training/trainer.py
├── schemas/
└── services/
└── main.py

text

## 🎮 ENVIRONMENTS (7 REQUIRED)
1. GridWorld (custom grid)
2. CartPole (Gymnasium)
3. MountainCar (Gymnasium) 
4. FrozenLake (Gymnasium)
5. Breakout (Gymnasium Atari)
6. Gym4Real (custom)
7. **EVERY env MUST implement:**
class BaseEnv:
def reset(self) -> dict
def step(self, action: int) -> tuple[dict, float, bool]
def render_state(self) -> dict # JSON for frontend

text

## 🧠 ALGORITHMS (7 REQUIRED)
1. Policy Iteration
2. Policy Evaluation  
3. Value Iteration
4. Monte Carlo (MC)
5. Temporal Difference (TD)
6. n-step TD
7. SARSA
8. Q-learning

**EVERY algo MUST implement:**
class BaseAgent:
def select_action(self, state: dict) -> int
def update(self, state: dict, action: int, reward: float, next_state: dict, done: bool)

text

## 🌐 API ENDPOINTS (MANDATORY)
GET /api/v1/envs # List all 7 envs
POST /api/v1/envs/{name}/reset
POST /api/v1/train/sessions # Start training
GET /api/v1/train/sessions/{id}
WS /ws/training/{session_id} # Real-time updates

text

## 🎨 FRONTEND (SINGLE PAGE)
frontend/src/
├── pages/MainDashboard.tsx # ONLY PAGE
├── components/
│ ├── layout/AppLayout.tsx # Sidebar + Main
│ ├── controls/ # Selectors + Sliders
│ └── visualization/ # 7 env views + charts
└── hooks/useWebSocket.ts # Real-time

text

**Tech:** React 18 + TypeScript + MUI + Recharts + Canvas

## 📊 VISUALIZATIONS (REAL-TIME)
- **Environment animation** (60fps Canvas)
- **Value function heatmap**
- **Policy arrows overlay** 
- **Reward curve chart**
- **Convergence plot**
- **Action histogram**

## ⚙️ USER CONTROLS
- Environment selector (7 options)
- Algorithm selector (8 options)
- Hyperparameter sliders: `gamma`, `alpha`, `epsilon`, `n_steps`
- Buttons: Start, Pause, Resume, Reset

## 🛠️ CODE CONVENTIONS (MANDATORY)

### Python (Backend - PEP 8 + Strict)
- **Naming:** Descriptive, intention-revealing names; `snake_case`; avoid single letters except `i/j` for loops
- **Functions:** <20 lines, single responsibility; ≤3 args max; prefer dataclasses/kwargs for complex inputs
- **Comments:** Avoid unless TODOs/complex algorithms; use type hints + docstrings
- **Lines:** ≤88 chars; 4-space indentation; blank lines separate logical sections
- **SOLID:** Single responsibility per class/function; `typing.Protocol` for interfaces
- **Control Flow:** Avoid deep nesting; early returns, guard clauses; prefer comprehensions
- **DRY:** Extract repeated logic; use generators for iterables
- **Refactor:** Boy Scout Rule—leave code cleaner; `black` + `ruff`

### TypeScript (Frontend)
- **Naming:** `camelCase` components, `PascalCase`; descriptive names
- **Components:** Single responsibility; <100 lines; prefer hooks over class components
- **Hooks:** Custom hooks for logic (`useTraining`, `useWebSocket`)
- **TypeScript:** Strict mode; exhaustive switch; no `any`

## 🚫 NEVER DO
- Use routing (single page only)
- Add Docker (local dev only)
- Create new folder structure
- Use external UI templates
- Skip `render_state()` method
- Add database (in-memory only)
- Use single-letter variables (except loop `i/j`)
- Write functions >20 lines
- Deeply nested conditionals (>3 levels)

## ✅ ALWAYS FOLLOW
- **Pure functions** in `domain/` (no web dependencies)
- **Pydantic schemas** for ALL API I/O
- **WebSocket streaming** for training updates
- **60fps Canvas** for environment animation
- **MUI components** for controls/charts
- **TypeScript** everywhere in frontend
- **Type hints** everywhere in backend
- **Black + Ruff** formatting

## 📏 FORMATTING TOOLS
Backend
pip install black ruff isort
black .
ruff check --fix
isort .

Frontend
npm install --save-dev prettier eslint @typescript-eslint/parser
npx prettier --write .

text

## 🎓 COURSE PROJECT FOCUS
**Prioritize:** RL algorithms + visualization  
**Minimize:** Framework complexity, styling, auth, deployment