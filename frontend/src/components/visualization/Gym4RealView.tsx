import { useEffect, useState } from "react"
import type { Gym4RealRenderState } from "../../types/environment"
import { Card } from "../ui/card"
import { Navigation, MapPin, Target, Gauge, CheckCircle2, Circle } from "lucide-react"

interface Gym4RealViewProps {
  state: Gym4RealRenderState
}

export function Gym4RealView({ state }: Gym4RealViewProps) {
  const position = state.position ?? [0, 0]
  const goal = state.goal ?? [5, 5]
  const movement_trail = state.movement_trail ?? []
  const last_action = state.last_action
  const step = state.step ?? 0
  const reward = state.reward ?? 0
  const done = state.done ?? false
  const maxSteps = state.max_steps ?? 50

  const [isMoving, setIsMoving] = useState(false)
  const [prevPosition, setPrevPosition] = useState(position)
  const [showSuccessBurst, setShowSuccessBurst] = useState(false)

  // Grid settings
  const gridMin = -2
  const gridMax = 10
  const gridRange = gridMax - gridMin
  const canvasSize = 320

  // Convert coordinates to canvas position
  const toCanvasX = (x: number) => ((x - gridMin) / gridRange) * canvasSize
  const toCanvasY = (y: number) => canvasSize - ((y - gridMin) / gridRange) * canvasSize

  const agentX = toCanvasX(position[0])
  const agentY = toCanvasY(position[1])
  const goalX = toCanvasX(goal[0])
  const goalY = toCanvasY(goal[1])

  // Distance calculation
  const dx = position[0] - goal[0]
  const dy = position[1] - goal[1]
  const distance = Math.sqrt(dx * dx + dy * dy)
  const maxDistance = Math.sqrt(50) // Max possible distance
  const proximityRadius = Math.max(20, (distance / maxDistance) * 60)

  // Direction rotation from action
  const getRotation = (action: number | null): number => {
    if (action === 0) return -90  // UP
    if (action === 1) return 0    // RIGHT
    if (action === 2) return 90   // DOWN
    if (action === 3) return 180  // LEFT
    return 0
  }

  // Progress calculation
  const progressPercent = (step / maxSteps) * 100
  const isLowSteps = step > maxSteps * 0.8

  // Get progress bar color
  const getProgressColor = () => {
    if (progressPercent > 80) return '#ef4444'
    if (progressPercent > 50) return '#eab308'
    return '#22c55e'
  }

  // Detect position changes for animation
  useEffect(() => {
    if (position[0] !== prevPosition[0] || position[1] !== prevPosition[1]) {
      setIsMoving(true)
      setTimeout(() => setIsMoving(false), 300)
      setPrevPosition(position)
    }
  }, [position, prevPosition])

  // Success burst when goal reached
  useEffect(() => {
    if (done && distance < 0.5) {
      setShowSuccessBurst(true)
      setTimeout(() => setShowSuccessBurst(false), 600)
    }
  }, [done, distance])

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Navigation className="h-5 w-5 text-indigo-500" />
            <h3 className="text-lg font-semibold text-foreground">Gym4Real - 2D Robot</h3>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {done ? (
              distance < 0.5 ? (
                <>
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">Goal Reached!</span>
                </>
              ) : (
                <>
                  <Circle className="h-4 w-4 text-red-500" />
                  <span className="text-red-600 font-medium">Out of Steps</span>
                </>
              )
            ) : (
              <>
                <Circle className="h-4 w-4 text-indigo-500" />
                <span>Running</span>
              </>
            )}
          </div>
        </div>

        {/* Step Progress Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Steps: {step}/{maxSteps}</span>
            <span>{Math.round(100 - progressPercent)}% remaining</span>
          </div>
          <div className={`h-2 bg-muted/50 rounded-full overflow-hidden ${isLowSteps ? 'step-progress-low' : ''}`}>
            <div 
              className="h-full transition-all duration-300 rounded-full"
              style={{ 
                width: `${progressPercent}%`,
                background: `linear-gradient(90deg, #22c55e 0%, #eab308 50%, ${getProgressColor()} 100%)`
              }}
            />
          </div>
        </div>

        {/* Enhanced Visualization */}
        <div className="gym4real-environment aspect-square rounded-xl border border-indigo-900/50 p-4 flex items-center justify-center">
          <svg width="100%" height="100%" viewBox={`0 0 ${canvasSize} ${canvasSize}`} className="max-w-full">
            <defs>
              {/* Grid gradient */}
              <linearGradient id="gridGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#6366f1" stopOpacity="0.2" />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.1" />
              </linearGradient>
              
              {/* Robot gradient */}
              <linearGradient id="robotGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" />
                <stop offset="100%" stopColor="#1d4ed8" />
              </linearGradient>

              {/* Goal gradient */}
              <radialGradient id="goalGradient" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#fbbf24" />
                <stop offset="100%" stopColor="#d97706" />
              </radialGradient>

              {/* Trail gradient */}
              <linearGradient id="trailGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.2" />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.8" />
              </linearGradient>
            </defs>

            {/* Background */}
            <rect x="0" y="0" width={canvasSize} height={canvasSize} fill="url(#gridGradient)" rx="8" />

            {/* Grid Lines */}
            {Array.from({ length: 13 }, (_, i) => {
              const pos = (i / 12) * canvasSize
              const isMajor = i % 2 === 0
              return (
                <g key={i}>
                  <line
                    x1={pos}
                    y1={0}
                    x2={pos}
                    y2={canvasSize}
                    stroke={isMajor ? "rgba(129, 140, 248, 0.4)" : "rgba(99, 102, 241, 0.2)"}
                    strokeWidth={isMajor ? 1.5 : 1}
                  />
                  <line
                    x1={0}
                    y1={pos}
                    x2={canvasSize}
                    y2={pos}
                    stroke={isMajor ? "rgba(129, 140, 248, 0.4)" : "rgba(99, 102, 241, 0.2)"}
                    strokeWidth={isMajor ? 1.5 : 1}
                  />
                </g>
              )
            })}

            {/* Coordinate Labels */}
            {[-2, 0, 2, 4, 6, 8, 10].map((val) => {
              const x = toCanvasX(val)
              const y = toCanvasY(val)
              return (
                <g key={`label-${val}`}>
                  <text x={x} y={canvasSize - 5} fontSize="10" fill="rgba(165, 180, 252, 0.7)" textAnchor="middle">
                    {val}
            </text>
                  <text x={10} y={y + 3} fontSize="10" fill="rgba(165, 180, 252, 0.7)" textAnchor="start">
                    {val}
            </text>
                </g>
              )
            })}

            {/* Start Zone */}
            <rect
              x={toCanvasX(-0.5)}
              y={toCanvasY(0.5)}
              width={toCanvasX(0.5) - toCanvasX(-0.5)}
              height={toCanvasY(-0.5) - toCanvasY(0.5)}
              fill="rgba(34, 197, 94, 0.15)"
              stroke="rgba(34, 197, 94, 0.4)"
              strokeWidth="1"
              rx="4"
            />

            {/* Goal Zone */}
            <rect
              x={toCanvasX(4.5)}
              y={toCanvasY(5.5)}
              width={toCanvasX(5.5) - toCanvasX(4.5)}
              height={toCanvasY(4.5) - toCanvasY(5.5)}
              fill="rgba(251, 191, 36, 0.15)"
              stroke="rgba(251, 191, 36, 0.4)"
              strokeWidth="1"
              rx="4"
            />

            {/* Movement Trail */}
            {movement_trail.length > 1 && (
              <>
                {/* Trail line */}
                <polyline
                  points={movement_trail.map(([x, y]) => `${toCanvasX(x)},${toCanvasY(y)}`).join(' ')}
                  fill="none"
                  stroke="url(#trailGradient)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.7"
                />
                {/* Trail dots */}
                {movement_trail.map(([x, y], i) => {
                  const opacity = 0.2 + (i / movement_trail.length) * 0.6
                  const size = 3 + (i / movement_trail.length) * 3
                  return (
                    <circle
                      key={`trail-${i}`}
                      cx={toCanvasX(x)}
                      cy={toCanvasY(y)}
                      r={size}
                      fill="#22d3ee"
                      opacity={opacity}
                    />
                  )
                })}
              </>
            )}

            {/* Distance Line */}
            <line
              x1={agentX}
              y1={agentY}
              x2={goalX}
              y2={goalY}
              stroke="rgba(156, 163, 175, 0.4)"
              strokeWidth="1.5"
              strokeDasharray="6 4"
            />

            {/* Proximity Circle around Goal */}
            <circle
              cx={goalX}
              cy={goalY}
              r={proximityRadius}
              fill="none"
              stroke={distance < 2 ? "rgba(34, 197, 94, 0.5)" : "rgba(251, 191, 36, 0.3)"}
              strokeWidth="2"
              strokeDasharray="4 4"
            />

            {/* Goal Beacon */}
            <g className="goal-beacon">
              {/* Beacon rings */}
              <circle
                cx={goalX}
                cy={goalY}
                r="20"
                fill="none"
                stroke="rgba(251, 191, 36, 0.3)"
                strokeWidth="2"
                className="beacon-ring"
              />
              <circle
                cx={goalX}
                cy={goalY}
                r="20"
                fill="none"
                stroke="rgba(251, 191, 36, 0.3)"
                strokeWidth="2"
                className="beacon-ring-2"
              />
              {/* Goal circle */}
            <circle
              cx={goalX}
              cy={goalY}
                r="16"
                fill="url(#goalGradient)"
                stroke="#fbbf24"
                strokeWidth="2"
                filter="drop-shadow(0 0 8px rgba(251, 191, 36, 0.6))"
            />
              {/* Goal star */}
            <text
              x={goalX}
                y={goalY + 5}
                fontSize="16"
              textAnchor="middle"
              fill="white"
            >
              ⭐
            </text>
            </g>

            {/* Success Burst */}
            {showSuccessBurst && (
              <circle
                cx={goalX}
                cy={goalY}
                r="20"
                fill="rgba(34, 197, 94, 0.5)"
                className="goal-reached-burst"
              />
            )}

            {/* Enhanced Robot */}
            <g 
              transform={`translate(${agentX}, ${agentY})`}
              className={`robot-body ${isMoving ? 'robot-moving' : ''}`}
            >
              {/* Robot shadow */}
              <ellipse
                cx="0"
                cy="20"
                rx="14"
                ry="4"
                fill="rgba(0, 0, 0, 0.3)"
                filter="blur(2px)"
              />

              {/* Robot body */}
              <rect
                x="-14"
                y="-12"
                width="28"
                height="24"
                rx="4"
                fill="url(#robotGradient)"
                stroke="#1e40af"
                strokeWidth="1.5"
                filter="drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))"
              />

              {/* Wheels */}
              <rect x="-16" y="-8" width="4" height="8" rx="2" fill="#1f2937" stroke="#4b5563" strokeWidth="0.5" />
              <rect x="12" y="-8" width="4" height="8" rx="2" fill="#1f2937" stroke="#4b5563" strokeWidth="0.5" />
              <rect x="-16" y="4" width="4" height="8" rx="2" fill="#1f2937" stroke="#4b5563" strokeWidth="0.5" />
              <rect x="12" y="4" width="4" height="8" rx="2" fill="#1f2937" stroke="#4b5563" strokeWidth="0.5" />

              {/* Sensor dome */}
              <circle
                cx="0"
                cy="-4"
                r="6"
                fill="#22d3ee"
                stroke="#06b6d4"
                strokeWidth="1"
                className="robot-sensor"
                filter="drop-shadow(0 0 4px rgba(34, 211, 238, 0.5))"
              />

              {/* Antenna */}
              <line x1="0" y1="-12" x2="0" y2="-18" stroke="#6b7280" strokeWidth="2" />
              <circle
                cx="0"
                cy="-20"
                r="3"
                className="robot-antenna"
                fill="#ef4444"
              />

              {/* Direction indicator */}
              {last_action !== null && (
                <g transform={`rotate(${getRotation(last_action)})`}>
                  <polygon
                    points="0,-26 -5,-20 5,-20"
                    className="direction-arrow"
                    fill="#fbbf24"
                    stroke="#d97706"
                    strokeWidth="0.5"
                    filter="drop-shadow(0 0 4px rgba(251, 191, 36, 0.5))"
                  />
                </g>
              )}

              {/* Status LED */}
            <circle
                cx="6"
                cy="6"
                r="2"
                fill={done ? (distance < 0.5 ? "#22c55e" : "#ef4444") : "#3b82f6"}
                filter={`drop-shadow(0 0 3px ${done ? (distance < 0.5 ? "rgba(34, 197, 94, 0.8)" : "rgba(239, 68, 68, 0.8)") : "rgba(59, 130, 246, 0.8)"})`}
              />
            </g>

            {/* Distance label */}
            <g transform={`translate(${(agentX + goalX) / 2}, ${(agentY + goalY) / 2 - 10})`}>
              <rect
                x="-20"
                y="-8"
                width="40"
                height="16"
                rx="4"
                fill="rgba(0, 0, 0, 0.6)"
            />
            <text
                x="0"
                y="4"
              fontSize="10"
              textAnchor="middle"
              fill="white"
                fontFamily="monospace"
            >
                {distance.toFixed(1)}
            </text>
            </g>
          </svg>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <MapPin className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Position</p>
            </div>
            <p className="text-lg font-semibold text-foreground font-mono">
              [{position[0].toFixed(1)}, {position[1].toFixed(1)}]
            </p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Goal</p>
            </div>
            <p className="text-lg font-semibold text-foreground font-mono">
              [{goal[0].toFixed(1)}, {goal[1].toFixed(1)}]
            </p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Navigation className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Distance</p>
            </div>
            <p className={`text-lg font-semibold font-mono ${
              distance < 1 ? 'text-green-600' : distance < 3 ? 'text-yellow-600' : 'text-foreground'
            }`}>
              {distance.toFixed(2)}
            </p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Gauge className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Reward</p>
            </div>
            <p className={`text-lg font-semibold font-mono ${
              reward > 0 ? 'text-green-600' : 'text-foreground'
            }`}>
              {reward.toFixed(3)}
            </p>
          </Card>
        </div>
      </div>
    </Card>
  )
}
