import type { BreakoutRenderState } from "../../types/environment"
import { Card } from "../ui/card"
import { Gamepad2, Trophy, Heart, Activity, AlertCircle } from "lucide-react"

interface BreakoutViewProps {
  state: BreakoutRenderState
}

export function BreakoutView({ state }: BreakoutViewProps) {
  const score = state.score ?? 0
  const lives = state.lives ?? 5
  const step = state.step ?? 0
  const done = state.done ?? false
  const lastReward = state.last_reward ?? 0

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Gamepad2 className="h-5 w-5 text-purple-500" />
            <h3 className="text-lg font-semibold text-foreground">Breakout</h3>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {done ? (
              <>
                <AlertCircle className="h-4 w-4 text-red-500" />
                <span className="text-red-600">Game Over</span>
              </>
            ) : (
              <>
                <Activity className="h-4 w-4 text-green-500" />
                <span className="text-green-600">Playing</span>
              </>
            )}
          </div>
        </div>

        <div className="aspect-video bg-gradient-to-b from-background to-muted/30 rounded-xl border border-border p-6 flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center space-y-2">
              {done ? (
                <>
                  <AlertCircle className="h-12 w-12 mx-auto text-red-500" />
                  <p className="text-xl font-bold text-red-600">GAME OVER</p>
                </>
              ) : (
                <>
                  <Gamepad2 className="h-12 w-12 mx-auto text-green-500" />
                  <p className="text-lg font-semibold text-green-600">🎮 PLAYING</p>
                </>
              )}
            </div>
          </div>

          <div className="absolute top-4 left-4 right-4 flex justify-between items-start z-10">
            <div>
              <p className="text-xs text-muted-foreground mb-1">SCORE</p>
              <div className="flex items-center gap-2">
                <Trophy className="h-5 w-5 text-yellow-500" />
                <p className="text-2xl font-bold text-foreground">{score}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-muted-foreground mb-1">LIVES</p>
              <div className="flex items-center gap-1">
                {Array.from({ length: Math.max(0, lives) }).map((_, i) => (
                  <Heart key={i} className="h-5 w-5 fill-red-500 text-red-500" />
                ))}
              </div>
            </div>
          </div>

          <div className="absolute bottom-4 left-4 right-4 flex justify-between text-xs text-muted-foreground z-10">
            <span>STEP: {step}</span>
            <span>REWARD: {lastReward.toFixed(1)}</span>
          </div>

          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex gap-2 opacity-30">
            {["#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5", "#8e24aa"].map(
              (color, i) => (
                <div
                  key={i}
                  className="w-10 h-3 rounded"
                  style={{ backgroundColor: color }}
                />
              )
            )}
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Trophy className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Score</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{score}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Heart className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Lives</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{lives}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Steps</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{step}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              {done ? (
                <AlertCircle className="h-4 w-4 text-red-500" />
              ) : (
                <Activity className="h-4 w-4 text-green-500" />
              )}
              <p className="text-xs text-muted-foreground">Status</p>
            </div>
            <p className={`text-lg font-semibold ${done ? "text-red-600" : "text-green-600"}`}>
              {done ? "Over" : "Active"}
            </p>
          </Card>
        </div>
      </div>
    </Card>
  )
}
