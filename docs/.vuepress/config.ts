import { defineUserConfig } from 'vuepress'
import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'

export default defineUserConfig({
  title: 'CNN 速成课',
  description: '项目驱动的卷积神经网络入门',
  lang: 'zh-CN',
  bundler: viteBundler({
    viteOptions: {
      resolve: { preserveSymlinks: true }
    }
  }),
  theme: defaultTheme({
    repo: 'cnn_fast_onboarding',
    docsRepo: 'cnn_fast_onboarding',
    repoLabel: 'GitHub',
    docsDir: 'docs',
    editLink: false,
    sidebar: 'heading',
    navbar: [
      { text: '简介', link: '/' },
      { text: 'Ch 1', link: '/ch1/' },
      { text: 'Ch 2', link: '/ch2/' },
      { text: 'Ch 3', link: '/ch3/' },
      { text: 'Ch 4', link: '/ch4/' }
    ]
  }),
  markdown: {
    lineNumbers: true,
    anchor: { permalink: false }
  }
})
