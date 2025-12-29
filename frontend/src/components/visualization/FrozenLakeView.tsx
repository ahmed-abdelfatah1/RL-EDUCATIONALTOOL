import { useEffect, useState } from "react"
import type { FrozenLakeRenderState } from "../../types/environment"
import { Card } from "../ui/card"
import { Snowflake, MapPin, Target, CheckCircle2, Circle } from "lucide-react"

interface FrozenLakeViewProps {
  state: FrozenLakeRenderState
}

export function FrozenLakeView({ state }: FrozenLakeViewProps) {
  const agent_pos = state.agent_pos ?? [0, 0]
  const tiles = state.tiles ?? []
  const movement_trail = state.movement_trail ?? []
  const last_action = state.last_action
  const reward = state.reward ?? 0
  const done = state.done ?? false

  const [isSliding, setIsSliding] = useState(false)
  const [prevPosition, setPrevPosition] = useState(agent_pos)

  // Detect position changes for sliding animation
  useEffect(() => {
    if (agent_pos[0] !== prevPosition[0] || agent_pos[1] !== prevPosition[1]) {
      setIsSliding(true)
      setTimeout(() => setIsSliding(false), 300)
      setPrevPosition(agent_pos)
    }
  }, [agent_pos, prevPosition])

  // Get character direction from action
  const getCharacterDirection = (action: number | null): string => {
    if (action === 0) return 'facing-left'
    if (action === 1) return 'facing-down'
    if (action === 2) return 'facing-right'
    if (action === 3) return 'facing-up'
    return 'facing-right'
  }

  // Check if position is in trail
  const isInTrail = (row: number, col: number): boolean => {
    return movement_trail.some(([r, c]) => r === row && c === col)
  }

  // Get trail opacity based on recency
  const getTrailOpacity = (row: number, col: number): number => {
    const index = movement_trail.findIndex(([r, c]) => r === row && c === col)
    if (index === -1) return 0
    
    // More recent positions are more opaque
    const recency = (index + 1) / movement_trail.length
    return recency * 0.6
  }

  // Get tile CSS class based on tile type
  const getTileClass = (cell: string, isAgent: boolean, inTrail: boolean): string => {
    if (cell === "H") return "tile-hole"
    if (cell === "G") return "tile-goal"
    if (cell === "S") return "tile-start"
    return "tile-frozen"
  }

  if (tiles.length === 0) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur-sm">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-2">FrozenLake</h3>
          <p className="text-sm text-muted-foreground">Loading tiles...</p>
        </div>
      </Card>
    )
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Snowflake className="h-5 w-5 text-cyan-500" />
            <h3 className="text-lg font-semibold text-foreground">FrozenLake</h3>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {done ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <span className="text-green-600 font-medium">Episode Done</span>
              </>
            ) : (
              <>
                <Circle className="h-4 w-4 text-blue-500" />
                <span>Running</span>
              </>
            )}
          </div>
        </div>

        {/* Enhanced Visualization */}
        <div className="frozenlake-environment aspect-square rounded-xl border border-border p-8 flex items-center justify-center relative">
          {/* Snow Particles */}
          {[...Array(8)].map((_, i) => (
            <div
              key={`snow-${i}`}
              className="snow-particle"
              style={{
                width: `${3 + Math.random() * 3}px`,
                height: `${3 + Math.random() * 3}px`,
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 10}s`,
                animationDuration: `${8 + Math.random() * 4}s`,
              }}
            />
          ))}

          <div className="grid gap-2 w-full max-w-md relative" style={{ gridTemplateColumns: `repeat(${tiles[0]?.length || 4}, 1fr)` }}>
            {tiles.map((row, rowIndex) =>
              row.map((cell, colIndex) => {
                const isAgent = agent_pos[0] === rowIndex && agent_pos[1] === colIndex
                const inTrail = isInTrail(rowIndex, colIndex)
                const trailOpacity = getTrailOpacity(rowIndex, colIndex)
                const tileClass = getTileClass(cell, isAgent, inTrail)

                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={`aspect-square ${tileClass} transition-all relative`}
                  >
                    {/* Trail marker */}
                    {inTrail && !isAgent && (
                      <div className="skating-trail">
                        <div 
                          className="trail-mark absolute inset-2" 
                          style={{ opacity: trailOpacity }}
                        />
                      </div>
                    )}

                    {/* Hole ripple effect */}
                    {cell === "H" && (
                      <div className="hole-ripple" />
                    )}

                    {/* Goal sparkles */}
                    {cell === "G" && !isAgent && (
                      <>
                        <div className="goal-sparkle sparkle-1" />
                        <div className="goal-sparkle sparkle-2" />
                        <div className="goal-sparkle sparkle-3" />
                        <div className="absolute inset-0 flex items-center justify-center text-3xl">
                          ⭐
                        </div>
                      </>
                    )}

                    {/* Start flag */}
                    {cell === "S" && !isAgent && (
                      <div className="absolute inset-0 flex items-center justify-center text-2xl">
                        🏁
                      </div>
                    )}

                    {/* Animated Character Sprite */}
                    {isAgent && (
                      <div className={`character-sprite ${isSliding ? 'character-sliding' : ''}`}>
                        <div className={getCharacterDirection(last_action)}>
                          <div className="character-head">
                            <div className="character-eyes">
                              <div className="character-eye" />
                              <div className="character-eye" />
                            </div>
                            <div className="character-body" />
                            <div className="character-scarf" />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground bg-muted/30 rounded-lg p-3">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 tile-frozen rounded" />
            <span>Frozen Ice</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 tile-hole rounded" />
            <span>Hole</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 tile-goal rounded" />
            <span>Goal</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 tile-start rounded" />
            <span>Start</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 bg-blue-300/40 rounded" />
            <span>Trail</span>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-4">
          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <MapPin className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Agent Position</p>
            </div>
            <p className="text-lg font-semibold text-foreground">
              [{agent_pos[0]}, {agent_pos[1]}]
            </p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Reward</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{reward.toFixed(1)}</p>
          </Card>

          <Card className={`p-4 border-border ${
            done ? 'bg-green-500/10' : 'bg-blue-500/10'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {done ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <Circle className="h-4 w-4 text-blue-500" />
              )}
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
