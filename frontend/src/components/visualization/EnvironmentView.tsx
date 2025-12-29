import type {
  AnyEnvRenderState,
  GridWorldRenderState,
  MountainCarRenderState,
  CartPoleRenderState,
  FrozenLakeRenderState,
  Gym4RealRenderState,
} from "../../types/environment"
import { GridWorldView } from "./GridWorldView"
import { MountainCarView } from "./MountainCarView"
import { CartPoleView } from "./CartPoleView"
import { FrozenLakeView } from "./FrozenLakeView"
import { Gym4RealView } from "./Gym4RealView"
import { Card } from "../ui/card"
import { Grid3x3 } from "lucide-react"

interface EnvironmentViewProps {
  envName: string
  state: AnyEnvRenderState | null
}

export function EnvironmentView({ envName, state }: EnvironmentViewProps) {
  if (!state) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur-sm">
        <div className="p-6 text-center">
          <Grid3x3 className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
          <p className="text-lg font-semibold text-foreground mb-2">
            No state loaded for <strong>{envName}</strong>
          </p>
          <p className="text-sm text-muted-foreground">
            Select an environment to load its visualization.
          </p>
        </div>
      </Card>
    )
  }

  switch (envName) {
    case "gridworld":
      return <GridWorldView state={state as GridWorldRenderState} />

    case "mountaincar":
      return <MountainCarView state={state as MountainCarRenderState} />

    case "cartpole":
      return <CartPoleView state={state as CartPoleRenderState} />

    case "frozenlake":
      return <FrozenLakeView state={state as FrozenLakeRenderState} />

    case "gym4real":
      return <Gym4RealView state={state as Gym4RealRenderState} />

    default:
      return (
        <Card className="border-border bg-card/50 backdrop-blur-sm">
          <div className="p-6">
            <p className="text-foreground mb-2">
              Visualization not implemented for <strong>{envName}</strong>.
            </p>
            <pre className="text-xs overflow-auto bg-background p-4 rounded-lg border border-border">
              {JSON.stringify(state, null, 2)}
            </pre>
          </div>
        </Card>
      )
  }
}
