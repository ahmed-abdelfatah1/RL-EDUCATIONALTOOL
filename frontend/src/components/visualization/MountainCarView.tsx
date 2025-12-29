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
  const goalPosition = state.goal_position ?? 0.5
  const action = state.action ?? null

  const minPos = -1.2
  const maxPos = 0.6

  const calculateHeight = (pos: number): number => {
    return Math.sin(3 * pos) * 0.45 + 0.55
  }

  const calculateSlope = (pos: number): number => {
    return 3 * Math.cos(3 * pos) * 0.45
  }

  const posToX = (pos: number): number => {
    const ratio = (pos - minPos) / (maxPos - minPos)
    return Math.min(400, Math.max(0, ratio * 400))
  }

  const heightToY = (height: number): number => {
    return 60 - height * 40
  }

  const carX = posToX(position)
  const carHeight = calculateHeight(position)
  const carY = heightToY(carHeight) - 8
  const slope = calculateSlope(position)
  const tiltAngle = Math.atan(slope) * (180 / Math.PI) * 0.8

  const goalX = posToX(goalPosition)
  const goalHeight = calculateHeight(goalPosition)
  const goalY = heightToY(goalHeight)

  const generateMountainPath = (): string => {
    const points: string[] = []
    for (let x = 0; x <= 100; x += 1) {
      const pos = minPos + (x / 100) * (maxPos - minPos)
      const height = calculateHeight(pos)
      const yCoord = heightToY(height)
      points.push(`${x * 4},${yCoord}`)
    }
    return `M0,80 L0,${heightToY(calculateHeight(minPos))} L${points.join(" L")} L400,80 Z`
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

        <div className="aspect-video bg-background rounded-xl border border-border overflow-hidden">
          <svg width="100%" height="100%" viewBox="0 0 400 80" preserveAspectRatio="xMidYMid meet">
            <defs>
              <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="oklch(0.7 0.12 230)" />
                <stop offset="100%" stopColor="oklch(0.85 0.06 200)" />
              </linearGradient>
              <linearGradient id="hillGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="oklch(0.55 0.18 145)" />
                <stop offset="100%" stopColor="oklch(0.35 0.12 140)" />
              </linearGradient>
              <radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="oklch(0.98 0.12 90)" />
                <stop offset="100%" stopColor="oklch(0.95 0.15 85)" stopOpacity="0" />
              </radialGradient>
            </defs>

            <rect width="400" height="80" fill="url(#skyGradient)" />

            <circle cx="360" cy="12" r="18" fill="url(#sunGlow)" />
            <circle cx="360" cy="12" r="8" fill="oklch(0.98 0.15 90)" />

            <path d={generateMountainPath()} fill="url(#hillGradient)" />

            <g transform={`translate(${goalX}, ${goalY - 20})`}>
              <line x1="0" y1="0" x2="0" y2="20" stroke="oklch(0.3 0.05 145)" strokeWidth="2" />
              <polygon points="0,0 12,4 0,8" fill="oklch(0.6 0.25 30)" />
              <polygon points="0,0 12,4 0,8" fill="oklch(0.55 0.2 25)" opacity="0.8" />
            </g>

            <g transform={`translate(${carX}, ${carY}) rotate(${-tiltAngle})`}>
              {action === 0 && (
                <g transform="translate(14, 2)">
                  <polygon points="0,-2 8,0 0,2" fill="oklch(0.7 0.22 60)" opacity="0.9" />
                  <polygon points="4,-1 10,0 4,1" fill="oklch(0.8 0.18 55)" opacity="0.7" />
                </g>
              )}
              {action === 2 && (
                <g transform="translate(-14, 2)">
                  <polygon points="0,-2 -8,0 0,2" fill="oklch(0.7 0.22 60)" opacity="0.9" />
                  <polygon points="-4,-1 -10,0 -4,1" fill="oklch(0.8 0.18 55)" opacity="0.7" />
                </g>
              )}

              <rect
                x="-14"
                y="-6"
                width="28"
                height="12"
                rx="3"
                fill={done ? "oklch(0.55 0.22 145)" : "oklch(0.5 0.22 250)"}
              />
              <rect
                x="-12"
                y="-4"
                width="24"
                height="6"
                rx="2"
                fill={done ? "oklch(0.6 0.18 150)" : "oklch(0.6 0.18 245)"}
                opacity="0.6"
              />

              <circle cx="-7" cy="6" r="4" fill="oklch(0.15 0.01 265)" />
              <circle cx="-7" cy="6" r="2.5" fill="oklch(0.25 0.01 265)" />
              <line
                x1="-7"
                y1="4"
                x2="-7"
                y2="8"
                stroke="oklch(0.4 0.01 265)"
                strokeWidth="0.8"
                transform={`rotate(${velocity * 2000}, -7, 6)`}
              />

              <circle cx="7" cy="6" r="4" fill="oklch(0.15 0.01 265)" />
              <circle cx="7" cy="6" r="2.5" fill="oklch(0.25 0.01 265)" />
              <line
                x1="7"
                y1="4"
                x2="7"
                y2="8"
                stroke="oklch(0.4 0.01 265)"
                strokeWidth="0.8"
                transform={`rotate(${velocity * 2000}, 7, 6)`}
              />
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
