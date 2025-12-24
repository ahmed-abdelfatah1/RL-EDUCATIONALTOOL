import type { MountainCarRenderState } from "../../types/environment"
import { Card } from "../ui/card"
import { Mountain, Gauge, Target, CheckCircle2, Circle } from "lucide-react"

interface MountainCarViewProps {
  state: MountainCarRenderState
}

export function MountainCarView({ state }: MountainCarViewProps) {
  const position = state.position ?? -0.5
  const velocity = state.velocity ?? 0
  const reward = state.reward ?? 0
  const done = state.done ?? false
  const goal_position = state.goal_position ?? 0.5

  const minPos = -1.2
  const maxPos = 0.6
  const ratio = (position - minPos) / (maxPos - minPos)
  const leftPercent = Math.min(100, Math.max(0, ratio * 100))

  const goalRatio = (goal_position - minPos) / (maxPos - minPos)
  const goalPercent = Math.min(100, Math.max(0, goalRatio * 100))

  const generateMountainPath = () => {
    const points: string[] = []
    for (let x = 0; x <= 100; x += 2) {
      const pos = minPos + (x / 100) * (maxPos - minPos)
      const height = Math.sin(3 * pos) * 0.45 + 0.55
      const y = 60 - height * 40
      points.push(`${x * 4},${y}`)
    }
    return `M0,60 L${points.join(" L")} L400,60 Z`
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mountain className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-foreground">MountainCar</h3>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {done ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <span className="text-green-600">Goal Reached</span>
              </>
            ) : (
              <>
                <Circle className="h-4 w-4 text-blue-500" />
                <span>Running</span>
              </>
            )}
          </div>
        </div>

        <div className="aspect-video bg-background rounded-xl border border-border p-4 flex items-center justify-center">
          <svg width="100%" height="100%" viewBox="0 0 400 80" className="max-w-full">
            <path d={generateMountainPath()} fill="oklch(0.7 0.1 45)" />

            <g transform={`translate(${goalPercent * 4}, 15)`}>
              <line x1="0" y1="0" x2="0" y2="25" stroke="oklch(0.5 0.2 145)" strokeWidth="2" />
              <polygon points="0,0 15,5 0,10" fill="oklch(0.5 0.2 145)" />
            </g>

            <g transform={`translate(${leftPercent * 4}, 35)`}>
              <rect
                x="-12"
                y="-8"
                width="24"
                height="16"
                rx="4"
                fill={done ? "oklch(0.5 0.2 145)" : "oklch(0.55 0.25 250)"}
              />
              <circle cx="-6" cy="8" r="4" fill="oklch(0.2 0.01 265)" />
              <circle cx="6" cy="8" r="4" fill="oklch(0.2 0.01 265)" />
            </g>
          </svg>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Position</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{position.toFixed(3)}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Gauge className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Velocity</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{velocity.toFixed(4)}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Reward</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{reward.toFixed(1)}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Status</p>
            </div>
            <p className={`text-lg font-semibold ${done ? "text-green-600" : "text-blue-600"}`}>
              {done ? "Done ✓" : "Running..."}
            </p>
          </Card>
        </div>
      </div>
    </Card>
  )
}
