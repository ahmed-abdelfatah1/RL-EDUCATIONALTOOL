import { Component, ReactNode } from "react"
import { Button } from "./ui/button"
import { Card } from "./ui/card"
import { AlertCircle } from "lucide-react"

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack: string }) {
    console.error("React Error:", error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-6">
          <Card className="border-destructive/50 bg-card/50 backdrop-blur-sm max-w-2xl w-full">
            <div className="p-6 space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-destructive/20">
                  <AlertCircle className="h-6 w-6 text-destructive" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-foreground">Something went wrong</h2>
                  <p className="text-sm text-muted-foreground">An error occurred in the application</p>
                </div>
              </div>

              {this.state.error && (
                <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                  <pre className="text-sm text-destructive-foreground whitespace-pre-wrap break-words">
                    {this.state.error.message}
                  </pre>
                </div>
              )}

              <Button
                onClick={() => this.setState({ hasError: false, error: null })}
                className="w-full"
              >
                Try Again
              </Button>
            </div>
          </Card>
        </div>
      )
    }

    return this.props.children
  }
}
