import { useEffect, useState } from "react"
import type { GridWorldRenderState } from "../../types/environment"
import { Card } from "../ui/card"

interface GridWorldViewProps {
  state: GridWorldRenderState
}

export function GridWorldView({ state }: GridWorldViewProps) {
  const grid = state.grid ?? []
  const agent_pos = state.agent_pos ?? [0, 0]
  const goal_pos = state.goal_pos ?? [4, 4]
  const agent_direction = state.agent_direction ?? 2
  const movement_trail = state.movement_trail ?? []
  const last_reward = state.last_reward ?? 0

  const [rewardEffect, setRewardEffect] = useState<'positive' | 'negative' | null>(null)
  const [prevReward, setPrevReward] = useState(0)

  // Trigger reward effects
  useEffect(() => {
    if (last_reward !== prevReward) {
      if (last_reward > 0) {
        setRewardEffect('positive')
      } else if (last_reward < 0) {
        setRewardEffect('negative')
      }
      setPrevReward(last_reward)

      // Clear effect after animation
      const timer = setTimeout(() => setRewardEffect(null), 600)
      return () => clearTimeout(timer)
    }
  }, [last_reward, prevReward])

  if (grid.length === 0) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur-sm">
        <div className="p-6">
          <h2 className="text-lg font-semibold text-foreground mb-2">GridWorld</h2>
          <p className="text-sm text-muted-foreground">Loading grid...</p>
        </div>
      </Card>
    )
  }

  const getDirectionClass = (direction: number): string => {
    const directions = ['direction-up', 'direction-right', 'direction-down', 'direction-left']
    return directions[direction] || 'direction-down'
  }

  const isInTrail = (row: number, col: number): boolean => {
    return movement_trail.some(([r, c]) => r === row && c === col)
  }

  const getTrailOpacity = (row: number, col: number): number => {
    const index = movement_trail.findIndex(([r, c]) => r === row && c === col)
    if (index === -1) return 0
    
    // More recent positions are more opaque
    const recency = (index + 1) / movement_trail.length
    return recency * 0.6
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-foreground">GridWorld</h3>
          <div className="text-sm text-muted-foreground">
            Agent: [{agent_pos[0]}, {agent_pos[1]}] | Goal: [{goal_pos[0]}, {goal_pos[1]}]
          </div>
        </div>

        <div className="aspect-square bg-gradient-to-br from-green-900/20 to-green-950/30 rounded-xl border border-border p-8 flex items-center justify-center">
          <div className="grid grid-cols-5 gap-2 w-full max-w-md relative">
            {grid.map((row, rowIndex) =>
              row.map((_, colIndex) => {
                const isAgent = agent_pos[0] === rowIndex && agent_pos[1] === colIndex
                const isGoal = goal_pos[0] === rowIndex && goal_pos[1] === colIndex
                const inTrail = isInTrail(rowIndex, colIndex)
                const trailOpacity = getTrailOpacity(rowIndex, colIndex)

                let tileClass = "gridworld-tile tile-grass"
                if (isGoal) {
                  tileClass = "gridworld-tile tile-goal"
                } else if (inTrail && !isAgent) {
                  tileClass = "gridworld-tile tile-path"
                }

                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={`${tileClass} ${
                      isAgent && rewardEffect === 'positive' ? 'reward-positive' : ''
                    } ${
                      isAgent && rewardEffect === 'negative' ? 'reward-negative' : ''
                    }`}
                  >
                    {/* Trail marker */}
                    {inTrail && !isAgent && !isGoal && (
                      <div className="trail-marker">
                        <div 
                          className="trail-dot" 
                          style={{ opacity: trailOpacity }}
                        />
                      </div>
                    )}

                    {/* Goal star */}
                    {isGoal && !isAgent && (
                      <div className="absolute inset-0 flex items-center justify-center text-4xl">
                        ⭐
                      </div>
                    )}

                    {/* Agent sprite */}
                    {isAgent && (
                      <div className="gridworld-agent">
                        <div className={`agent-sprite ${getDirectionClass(agent_direction)}`} />
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Status info */}
        <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-br from-blue-500 to-blue-600" />
              <span>Agent</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-gradient-to-br from-amber-400 to-amber-600" />
              <span>Goal</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-gradient-to-br from-green-400 to-green-600" />
              <span>Path</span>
            </div>
          </div>
          {last_reward !== 0 && (
            <div className={`font-medium ${last_reward > 0 ? 'text-green-500' : 'text-red-500'}`}>
              {last_reward > 0 ? '+' : ''}{last_reward}
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}
