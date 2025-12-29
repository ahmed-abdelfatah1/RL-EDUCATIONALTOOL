import type { EpisodeMetrics } from "../../api/trainingApi"
import { Card } from "../ui/card"
import { Activity, TrendingUp } from "lucide-react"

interface RewardChartProps {
  episodes: EpisodeMetrics[]
}

export function RewardChart({ episodes }: RewardChartProps) {
  if (episodes.length === 0) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur-sm">
        <div className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-foreground">Total Reward per Episode</h3>
          </div>

          <div className="aspect-video bg-background rounded-xl border border-border flex items-center justify-center">
            <div className="text-center space-y-2">
              <Activity className="h-12 w-12 mx-auto text-muted-foreground opacity-50" />
              <p className="text-sm text-muted-foreground max-w-xs">
                No training data yet. Run training to see results and visualize performance metrics.
              </p>
            </div>
          </div>
        </div>
      </Card>
    )
  }

  const maxReward = Math.max(...episodes.map((ep) => ep.total_reward))
  const minReward = Math.min(...episodes.map((ep) => ep.total_reward))
  const range = maxReward - minReward || 1
  const avgReward = episodes.reduce((sum, ep) => sum + ep.total_reward, 0) / episodes.length

  return (
    <div className="space-y-6">
      <Card className="border-border bg-card/50 backdrop-blur-sm">
        <div className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-foreground">Total Reward per Episode</h3>
          </div>

          <div className="aspect-video bg-background rounded-xl border border-border p-4 flex items-end gap-1">
            {episodes.map((ep) => {
              const height = ((ep.total_reward - minReward) / range) * 100 + 10
              return (
                <div
                  key={ep.episode}
                  title={`Episode ${ep.episode}: ${ep.total_reward.toFixed(2)}`}
                  className="flex-1 min-w-[4px] max-w-[20px] bg-gradient-to-t from-blue-600 to-cyan-500 rounded-t transition-all hover:opacity-80"
                  style={{ height: `${Math.max(height, 5)}%` }}
                />
              )
            })}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-3 gap-4">
        <Card className="border-border bg-card/50 backdrop-blur-sm p-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Avg Reward</p>
            <p className="text-2xl font-bold text-foreground">{avgReward.toFixed(2)}</p>
          </div>
        </Card>
        <Card className="border-border bg-card/50 backdrop-blur-sm p-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Episodes</p>
            <p className="text-2xl font-bold text-foreground">{episodes.length}</p>
          </div>
        </Card>
        <Card className="border-border bg-card/50 backdrop-blur-sm p-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Max Reward</p>
            <p className="text-2xl font-bold text-foreground">{maxReward.toFixed(2)}</p>
          </div>
        </Card>
      </div>
    </div>
  )
}
